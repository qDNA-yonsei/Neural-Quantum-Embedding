# Neural Quantum Embedding

This is a code for [Neural Quantum Embedding: Pushing the Limits of Quantum Supervised Learning](https://arxiv.org/pdf/2311.11412).

1. *data.py*: Loads datasets. MNIST, Fashion-MNIST, and KMNIST data are used with various feature reduction methods.
2. *embedding.py*: Includes code for *ZZ feature embedding* and *QCNN (Quantum Convolutional Neural Networks)*.
3. *Hybrid_nn.py*: Code for various *Neural Quantum Embedding (NQE)* models.
4. *training.py*: Code for training various NQE models.
5. *QCNN_demonstration/*: Directory that contains results for QCNN experiments with and without NQE to demonstrate the effectiveness of NQE in QML performances. Noiseless, Noisy (simulation), and Real (IBM Quantum Hardwares) results are contained.
6. *Other Experiments/*: Directory that contains results for QCNN experiments with and without NQE on different datasets (Fashion-MNIST and KMNIST). Noiseless and Noisy (simulation) results are contained.

[![DOI](https://zenodo.org/badge/787288732.svg)](https://zenodo.org/doi/10.5281/zenodo.12817964)
