<h1>Prototype of Federated Learning with Multi-Key Homomorphic Encryption</h1>
<p>This repository contains an implementation that integrates the <a href="https://arxiv.org/abs/2104.06824">xMK-CKKS</a> homomorphic encryption scheme into a federated learning architecture to provide a demonstration of how to incorporate Multi-Key Homomorphic Encryption (MKHE) into a federated learning system.</p>

<h2>Dependencies</h2>
<p>This project is built upon two external repositories:</p>
<ul>
<li><a href="https://github.com/MetisPrometheus/rlwe-xmkckks"><code>rlwe-xmkckks</code></a> (Modified RLWE for xMK-CKKS)</li>
<li><a href="https://github.com/MetisPrometheus/flower-xmkckks"><code>flower-xmkckks</code></a> (Modified Flower library to incorporate MKHE using <a href="https://github.com/MetisPrometheus/rlwe-xmkckks"><code>rlwe-xmkckks</code></a>)</li>
</ul>
<p>To install these packages locally, clone the repositories, navigate into their respective directories and execute <code>pip install .</code></p>
<p>In addition to the above packages, this project requires certain Python packages, which can be installed using the <code>requirements.txt</code> file. Run the following command to install these dependencies:</p>
<pre><code>pip install -r requirements.txt</code></pre>

<h2>Hardware Requirements</h2>
<p>The project was created and tested on AWS EC2 instances. For setup and light tasks, a <a href="https://aws.amazon.com/ec2/instance-types/t3/"><code>t3.large</code></a> instance was sufficient. However, for more intensive computations like model training, you may require more powerful instances like <a href="https://aws.amazon.com/ec2/instance-types/c5/"><code>c5a.8xlarge</code></a> or <a href="https://aws.amazon.com/ec2/instance-types/g4/"><code>g4dn.4xlarge</code></a>.</p>

<h2>Running the Code</h2>
<p>To run the federated learning system:</p>
<ol>
<li>Clone this repository</li>
<li>Execute <code>python server.py</code> in a terminal to launch the server. You can define the number of clients in the <code>server.py</code> script.</li>
<li>In separate terminals (corresponding to the number of clients), run <code>python client.py</code>.</li>
</ol>

<h2>Data</h2>
<p>Our prototype uses PNG images related to COVID-19. To use your own data, adapt the <code>load_covid.py</code> script to your specific data format and needs.</p>

<h2>Known Limitations</h2>
<p>Detailed limitations and proposed future directions are discussed in the associated master's thesis, mainly focusing on potential enhancements to the <a href="https://github.com/MetisPrometheus/rlwe-xmkckks"><code>rlwe-xmkckks</code></a> and <a href="https://github.com/MetisPrometheus/flower-xmkckks"><code>flower-xmkckks</code></a> repositories rather than the files in this master repository.</p>

<h2>Contributions</h2>
<p>This prototype primarily aims to demonstrate the integration of the xMK-CKKS scheme into the Flower library for federated learning. For further enhancements, feel free to fork our modified Flower library. Currently, we are not seeking direct contributions to this repository.</p>

<h2>Files Overview</h2>
<p>Here's a brief overview of the major files in the repository:</p>
<ul>
<li><code>data</code>: This folder contains PNG x-ray images related to COVID-19 used for the prototype.</li>
<li><code>client.py</code>: This script creates a client for the federated learning environment. It covers functionalities like RLWE instance initialization, model training/evaluation, public key generation, data encryption, and decryption.</li>
<li><code>server.py</code>: This script sets up the server for the federated learning environment. The number of clients is defined within this script.</li>
<li><code>utils.py</code>: This file contains helper functions for model parameter management, padding and unpadding lists, finding prime numbers, and more.</li>
<li><code>cnn.py</code>: This script builds and compiles a CNN model using the TensorFlow Keras API.</li>
<li><code>local_noFL.py</code>: This script provides a local baseline for testing a local model without federated learning or encryption.</li>
<li><code>test_xmkckks.py</code>: This script facilitates the testing of the multi-key homomorphic encryption scheme, simulating client behavior locally without incorporating the federated learning library.</li>
<li><code>__init__.py</code>: This file allows the execution of <code>client.py</code> and <code>server.py</code> scripts from a terminal outside the main directory.</li>
<li><code>requirements.txt</code>: This file lists the Python dependencies required for the project.</li>
</ul>
<p>For detailed insights into each file and their functionalities, please refer to the comments within each file.</p>
