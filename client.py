# Standard Libraries
import importlib
import math
import os
import sys
import time
import warnings
from typing import List, Tuple

# Third Party Imports
import flwr as fl
import tensorflow as tf
from memory_profiler import memory_usage
from rlwe_xmkckks import RLWE, Rq
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Local Imports
from load_covid import *
# Get absolute paths to let a user run the script from anywhere
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.basename(current_directory)
working_directory = os.getcwd()
# Add parent directory to Python's module search path
sys.path.append(os.path.join(current_directory, '..'))
# Compare paths
if current_directory == working_directory:
    from cnn import CNN
    import utils
else:
    # Add current directory to Python's module search path
    CNN = importlib.import_module(f"{parent_directory}.cnn").CNN
    import utils


if __name__ == "__main__":
    # Load datasets and initiate memory usage
    X_train, X_test, y_train, y_test = load_raw_covid_data(limit=100)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1) #, random_state=1
    memory_usage_start = memory_usage()[0]

    # RLWE SETTINGS
    # hardcoded settings
    #n = 2**20  # power of 2
    #q = 100_000_000_003  # prime number, q = 1 (mod 2n)
    #t = 2_000_000_033  # prime number, t < q

    # dynamic settings
    WEIGHT_DECIMALS = 8
    model = CNN(WEIGHT_DECIMALS)
    utils.set_initial_params(model)
    params, _ = utils.get_flat_weights(model)
    print(params[0:20])

    # find closest 2^x larger than number of weights
    num_weights = len(params)
    n = 2 ** math.ceil(math.log2(num_weights))
    print(f"n: {n}")

    # decide value range t of plaintext
    max_weight_value = 10**WEIGHT_DECIMALS # 100_000_000 if full weights
    num_clients = 10
    t = utils.next_prime(num_clients * max_weight_value * 2) # 2_000_000_011
    print(f"t: {t}")

    # decide value range q of encrypted plaintext
    q = utils.next_prime(t * 50) # 100_000_000_567
    print(f"q: {q}")

    # create rlwe instance for this client
    std = 3  # standard deviation of Gaussian distribution
    rlwe = RLWE(n, q, t, std)



    class CnnClient(fl.client.NumPyClient):

        def __init__(self, rlwe_instance: RLWE, WEIGHT_DECIMALS: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.rlwe = rlwe_instance
            self.allpub = None
            self.model_shape = None
            self.model_length = None
            self.flat_params = None
            self.WEIGHT_DECIMALS = WEIGHT_DECIMALS

            self.model = CNN(WEIGHT_DECIMALS)
            utils.set_initial_params(self.model)

        def get_parameters(self, config):
            weights = utils.get_model_parameters(self.model)                                                                
            for w in weights:
                print("::::::::::::::::::::::::")
                print(w.shape) 
            return utils.get_model_parameters(self.model)

        # use this for default federated learning (changes in server.py also needed)
        """ def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(self.model, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(X_train, y_train, X_val, y_val, epochs=15)
            return utils.get_model_parameters(self.model), len(X_train), {} """
        
        # use this for lattice encrypted federated learning (changes in server.py also needed)
        def fit(self, parameters, config):  # type: ignore
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(X_train, y_train, X_val, y_val, epochs=15)
            return [], len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            # TODO: worth it?
            # receive flattened parameters from server here instead to allow for inter-round accuracy
            # unflatten weights then put into set_model_params
            utils.set_model_params(self.model, parameters)
            loss, accuracy = self.model.evaluate(X_test, y_test)

            return loss, len(X_test), {"accuracy": accuracy}
        
        ################################################################################################################
        #  Below steps are involved in the implementation of federated learning with multi-key homomorphic encryption  #
        ################################################################################################################

        def example_response(self, question: str, l: List[int]) -> Tuple[str, int]:
            response = "Here you go Alice!"
            answer = sum(l)
            return response, answer

        # Step 1) Server sends shared vector_a to clients and they all send back vector_b
        def generate_pubkey(self, vector_a: List[int]) -> List[int]:
            vector_a = self.rlwe.list_to_poly(vector_a, "q")
            self.rlwe.set_vector_a(vector_a)
            (_, pub) = rlwe.generate_keys()
            print(f"client pub: {pub}")
            return pub[0].poly_to_list()
        
        # Step 2) Server sends aggregated publickey allpub to clients and receive boolean confirmation
        def store_aggregated_pubkey(self, allpub: List[int]) -> bool:
            aggregated_pubkey = self.rlwe.list_to_poly(allpub, "q")
            self.allpub = (aggregated_pubkey, self.rlwe.get_vector_a())
            print(f"client allpub: {self.allpub}")
            return True


        # Step 3) After round, encrypt flat list of parameters into two lists (c0, c1)
        def encrypt_parameters(self, request) -> Tuple[List[int], List[int]]:
            print(f"request msg is: {request}")
            
            # Get nested model parameters and turn into long list
            flattened_weights, self.model_shape = utils.get_flat_weights(self.model)

            # Pad list until length 2**20 with random numbers that mimic the weights
            flattened_weights, self.model_length = utils.pad_to_power_of_2(flattened_weights, self.rlwe.n, self.WEIGHT_DECIMALS)
            print(f"Client old plaintext: {self.flat_params[925:935]}") if self.flat_params is not None else None
            print(f"Client new plaintext: {flattened_weights[925:935]}")
            # Turn list into polynomial
            poly_weights = Rq(np.array(flattened_weights), self.rlwe.t)
            # print(f"Client plainpoly: {poly_weights}")

            # get gradient instead of full weights
            if request == "gradient":
                gradient = list(np.array(flattened_weights) - np.array(self.flat_params))
                print(f"Client gradient: {gradient[925:935]}")
                poly_weights = Rq(np.array(gradient), self.rlwe.t)

            # Encrypt the polynomial
            c0, c1 = self.rlwe.encrypt(poly_weights, self.allpub)
            c0 = list(c0.poly.coeffs)
            c1 = list(c1.poly.coeffs)
            print(f"c0: {c0[:10]}")
            print(f"c1: {c1[:10]}")
            return c0, c1



        # Step 4) Use csum1 to calculate partial decryption share di
        def compute_decryption_share(self, csum1) -> List[int]:
            std = 5
            csum1_poly = self.rlwe.list_to_poly(csum1, "q")
            error = Rq(np.round(std * np.random.randn(n)), q)
            d1 = self.rlwe.decrypt(csum1_poly, self.rlwe.s, error)
            d1 = list(d1.poly.coeffs) #d1 is poly_t not poly_q
            return d1



        # Step 5) Retrieve approximated model weights from server and set the new weights
        def receive_updated_weights(self, server_flat_weights) -> bool:
            # Convert list of python integers into list of np.float64
            server_flat_weights = list(np.array(server_flat_weights, dtype=np.float64))
            # self.flat_params = server_flat_weights

            if self.flat_params is None:
                # first round (server gives full weights)
                self.flat_params = server_flat_weights
            else:
                # next rounds (server gives only gradient)
                self.flat_params = list(np.array(self.flat_params) + np.array(server_flat_weights))
            
            # Remove padding and return weights to original tensor structure and set model weights
            server_flat_weights = self.flat_params[:self.model_length]
            # Restore the long list of weights into the neural network's original structure
            server_weights = utils.unflatten_weights(server_flat_weights, self.model_shape)
            print(f"Fedavg plaintext: {server_flat_weights[925:935]}")

            utils.set_model_params(self.model, server_weights)
            y_pred = self.model.model.predict(X_test)

            predicted = np.argmax(y_pred, axis=-1)
            accuracy = np.equal(y_test, predicted).mean()
            loss = log_loss(y_test, y_pred)

            precision = precision_score(y_test, predicted)
            recall = recall_score(y_test, predicted)
            f1_score_ = f1_score(y_test, predicted)
            confusion_matrix_ = confusion_matrix(y_test, predicted)
            
            y_pred = self.model.model.predict(X_val)
            predicted = np.argmax(y_pred, axis=-1)
            val_accuarcy = np.equal(y_val, predicted).mean()

            print()
            print(f"\nLen(X_test): {len(X_test)}")
            print(f"Accuracy: {accuracy}")
            print(f"Val Accuarcy: {val_accuarcy}")
            print("Precision:", precision)
            print("Recall:",recall)
            print("F1-Score:", f1_score_)
            print(f"Loss: {loss}")
            print("\nConfusion matrix")
            print(confusion_matrix_)
            print()
            
            return True



    
    fl.client.start_numpy_client(
        server_address="0.0.0.0:8080",
        client=CnnClient(rlwe, WEIGHT_DECIMALS)
    )

    # Stop recording memory usage
    memory_usage_end = memory_usage()[0]
    memory_usage_total = memory_usage_end - memory_usage_start
    print("Memory usage:", memory_usage_total, "MiB")