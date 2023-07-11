import sys
sys.path.insert(0, "/home/ubuntu/flower_master/src/py") # forked flwr
import flwr as fl
import pickle  # To serialize and deserialize Python objects
import socket
from typing import Dict, Optional
from sklearn.metrics import log_loss
from flwr.common import NDArrays, Scalar
from typing import List, Tuple, Optional, Dict, Union
sys.path.append('/home/ubuntu/cnn/FL')
import utils
print(fl.__path__)
from load_covid import *
sys.path.append('/home/ubuntu')
import cnn  # noqa
from flwr.server.strategy import FedAvg
from she import RLWE, Rq
import math

import time
from sklearn.metrics import classification_report
from memory_profiler import memory_usage


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in 'evaluate' itself
    _, X_test, _, y_test = load_raw_covid_data(limit=500)

    # The 'evaluate' function will be called after every round
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        utils.set_model_params(model, parameters)

        y_pred = model.model.predict(X_test)
        predicted = np.argmax(y_pred, axis=-1)
        accuracy = np.equal(y_test, predicted).mean()
        loss = log_loss(y_test, y_pred)

        print(classification_report(y_test, predicted))

        return loss, {"accuracy": accuracy}
    
    return evaluate



class CustomFedAvg(FedAvg):
    def __init__(self, rlwe_instance: RLWE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rlwe = rlwe_instance

    #def aggregate_fit(self, rnd: int, results: List[Tuple[str, CustomFitRes]], failures: List[str]) -> Tuple[List[np.ndarray], Dict[str, Scalar]]:
        # Use self.rlwe_instance for decryption and aggregation
        # pass



# Start Flower server for five rounds of federated learning
if __name__ == "__main__": 
    def measure_memory():
        # Measure the execution time
        start_time = time.time()

        # RLWE SETTINGS (dynamically)
        WEIGHT_DECIMALS = 8
        model = cnn.cnn.CNN(WEIGHT_DECIMALS)
        utils.set_initial_params(model)
        params, _ = utils.get_flat_weights(model)

        # find closest 2^x larger than number of weights
        num_weights = len(params)
        n = 2 ** math.ceil(math.log2(num_weights))
        print(f"n: {n}")

        # decide value range t of plaintext
        max_weight_value = 10**WEIGHT_DECIMALS # 100_000_000 if full weights
        num_clients = 2
        t = utils.next_prime(num_clients * max_weight_value * 2) # 2_000_000_011
        print(f"t: {t}")

        # decide value range q of encrypted plaintext
        q = utils.next_prime(t * 50) # *50 = 100_000_000_567
        print(f"q: {q}")

        std = 3 # standard deviation of Gaussian distribution
        rlwe = RLWE(n, q, t, std)
        




        # Custom Strategy
        strategy = CustomFedAvg(
            min_available_clients=2,
            min_fit_clients=2,
            evaluate_fn=get_evaluate_fn(model),
            on_fit_config_fn=fit_round,
            rlwe_instance=rlwe,
        )

        # Define Server and Client Manager
        client_manager = fl.server.SimpleClientManager()
        server = fl.server.Server(
            strategy=strategy,
            client_manager=client_manager
        )

        # Start Server
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            server=server,
            config=fl.server.ServerConfig(num_rounds=5),
        )

        # Calculate the execution time
        execution_time = time.time() - start_time

        # Print the execution time
        print("Execution time:", execution_time)

    mem_usage = memory_usage(measure_memory)
    print("Memory usage (in MB):", max(mem_usage))