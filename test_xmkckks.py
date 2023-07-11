# Standard Libraries
import os
import pickle
# Third Party Imports
import numpy as np
# Local Imports
from rlwe_xmkckks import RLWE, Rq



# Guassian distribution to add errors for the partial decryption
# Must have larger standard deviation than the errors used for encryption
def discrete_gaussian(n, q, mean=0., std=1.):
    coeffs = np.round(std * np.random.randn(n))
    return Rq(coeffs, q)

# q=67108289, t=1021
if __name__ == '__main__':
	# Hardcoded rlwe settings, check client.py or server.py for a dynamic option
	n = 2**20  # power of 2
	q = 100_000_000_003  # prime number, q = 1 (mod 2n)   
	t = 200_000_001  # prime number, t < q
	std = 3  # standard deviation of Gaussian distribution

	script_directory = os.path.dirname(os.path.realpath(__file__))
	file_path = os.path.join(script_directory, 'flatparameters.pkl')

	# Long list of weights to simulate how we convert nested tensors of weight into a long list
	flatparameters = []
	with open(file_path, 'rb') as f:
		flatparameters = pickle.load(f)
	print(n)
	print(flatparameters[925:935])

	# Give every client and instance their own instance with the same parameters 
	rlwe = RLWE(n, q, t, std)
	rlwe.generate_vector_a() # generate vector on server and use set_vector on clients

	# simulating 3 clients
	(sec1, pub1) = rlwe.generate_keys()
	(sec2, pub2) = rlwe.generate_keys()
	(sec3, pub3) = rlwe.generate_keys()
	print(f"sec1: {sec1}")
	print(f"sec2: {sec2}")
	print(f"sec3: {sec3}")

	# creating shared public key
	print(pub1[0])
	print(pub2[0])
	print(pub1[0] + pub2[0])
	allpub = (pub1[0] + pub2[0] + pub3[0], pub2[1])

	# plaintext on clients to be encrypted	
	m0 = Rq(np.random.randint(int(t/20), size=n), t)  # plaintext
	m1 = Rq(np.random.randint(int(t/20), size=n), t)  # plaintext
	m2 = Rq(np.random.randint(int(t/20), size=n), t)  # plaintext
	#m0 = Rq(np.array([300, 20, 300, 200, 400, 6, 7, 8]), t)  # plaintext
	#m1 = Rq(np.array([100, 2, 3, 4, 5, 6, 7, 400]), t)  # plaintext

	#conversion between python lists and polynomial objects
	#test = list(m0.poly.coeffs)
	#test = Rq(np.array(test), t)

	# encrypt plaintext m0 into (c0_0, c0_1)
	c0 = rlwe.encrypt(m0, allpub)
	c1 = rlwe.encrypt(m1, allpub)
	c2 = rlwe.encrypt(m2, allpub)

	# aggregate all cn_0 and cn_1
	csum0 = c0[0] + c1[0] + c2[0]
	csum1 = c0[1] + c1[1] + c2[1]
	csum = (csum0, csum1)
	print(f"csum0: {csum0}")
	print(f"csum1: {csum1}")

	# client 1 partial decryption
	e1star = discrete_gaussian(n, q, 5) # larger variance than error distr from public key generation (3)
	d1 = rlwe.decrypt(csum1, sec1, e1star)
	print(f"plain1: {m0}")
	print(f"d1: {d1}")
	print()

	# client 2 partial decryption
	e2star = discrete_gaussian(n, q, 5)
	d2 = rlwe.decrypt(csum1, sec2, e2star)
	print(f"plain2: {m1}")
	print(f"d2: {d2}")
	print()

	# client 3 partial decryption
	e3star = discrete_gaussian(n, q, 5)
	d3 = rlwe.decrypt(csum1, sec3, e3star)
	print(f"plain3: {m2}")
	print(f"d3: {d3}")
	print()

	# Let server decrypt the secure aggregation
	dec_sum = csum0 + d1 + d2 + d3
	dec_sum = Rq(dec_sum.poly.coeffs, t)
	print(f"plaintext sum: {m0 + m1 + m2}")
	print(f"decrypted sum: {dec_sum}")
	
	# testing decrypted sum if not all partial decryption shares are given
	dec_sum = csum0 + d1
	dec_sum = Rq(dec_sum.poly.coeffs, t)
	print(dec_sum)