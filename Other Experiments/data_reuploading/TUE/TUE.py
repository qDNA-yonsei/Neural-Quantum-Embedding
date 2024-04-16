import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/tak/Github/QEmbedding/')
import torch
from torch import nn
import data
import embedding
import pennylane as qml
from pennylane import numpy as np
import os

def train_embeddings(num_qubits, num_layers, num_data):
    PATH_DIR = f'nq:{num_qubits}_nl:{num_layers}_nd:{num_data}'
    dev = qml.device("default.qubit", wires=num_qubits)
    feature_reduction = f'PCA{num_qubits}'
    classes = [0, 1]
    X_, X_t, Y_, Y_t = data.data_load_and_process('mnist', feature_reduction=feature_reduction, classes=classes)
    X1_train, X0_train = [], []
    for i in range(len(X_)):
        if Y_[i] == 1:
            X1_train.append(X_[i])
        else:
            X0_train.append(X_[i])

    X1_test, X0_test = [], []
    for i in range(len(X_t)):
        if Y_t[i] == 1:
            X1_test.append(X_t[i])
        else:
            X0_test.append(X_t[i])

    X1_train, X0_train = X1_train[:num_data], X0_train[:num_data]
    X1_test, X0_test = X1_test[:num_data], X0_test[:num_data]
    Y_train = [1] * len(X1_train) + [0] * len(X0_train)
    Y_test = [1] * len(X1_test) + [0] * len(X0_test)
    X_train = X1_train + X0_train
    X_test = X1_test + X0_test
    X1_train, X0_train = torch.tensor(X1_train).to(torch.float32), torch.tensor(X0_train).to(torch.float32)
    X1_test, X0_test = torch.tensor(X1_test).to(torch.float32), torch.tensor(X0_test).to(torch.float32)

    def two_QuantumEmbedding1(x):
        for _ in range(3):
            for i in range(2):
                qml.Hadamard(wires=i)
                qml.RZ(-2 * x[i], wires=i)
            qml.CNOT(wires=[0,1])
            qml.RZ(-2 * (np.pi - x[0]) * (np.pi - x[1]), wires=1)
            qml.CNOT(wires=[0,1])

    def data_reuploading_ansatz2(params): #4 params
        for i in range(2):
            qml.RY(params[i], wires=i)
        for i in range(1):
            qml.IsingYY(params[i+2], wires=[i,i+1])
        qml.IsingYY(params[3], wires=[1,0])
    
    def data_reuploading_ansatz4(params): #8 params
        for i in range(4):
            qml.RY(params[i], wires=i)
        for i in range(3):
            qml.IsingYY(params[i+4], wires=[i,i+1])
        qml.IsingYY(params[7], wires=[3,0])

    def data_reuploading_embedding(weights, x):
        for l in range(num_layers):
            if num_qubits == 2:
                data_reuploading_ansatz2(weights[4 * l : 4 * (l + 1)])
                two_QuantumEmbedding1(x)
            elif num_qubits == 4:
                data_reuploading_ansatz4(weights[8 * l : 8 * (l + 1)])
                embedding.Four_QuantumEmbedding1(x)

    @qml.qnode(dev, interface="torch")
    def overlap_reuploading(inputs, weights): 
        data_reuploading_embedding(weights, inputs[0:num_qubits])
        qml.adjoint(data_reuploading_embedding)(weights, inputs[num_qubits:2 * num_qubits])
        return qml.probs(wires=range(num_qubits))
    
    class data_reuploading(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer2 = qml.qnn.TorchLayer(overlap_reuploading, weight_shapes={"weights":2 * num_qubits * num_layers})
        def forward(self, x1, x2):
            x = torch.concat([x1, x2], 1)
            x = self.qlayer2(x)
            return x[:,0]
        
    def new_data(X, Y):
        X1_new, X2_new, Y_new = [], [], []
        if num_data <= 3:
            for i in range(len(X)):
                for j in range(len(X)):
                    X1_new.append(X[i])
                    X2_new.append(X[j])
                    if Y[i] == Y[j]:
                        Y_new.append(1)
                    else:
                        Y_new.append(0)
        else:
            X1_new, X2_new, Y_new = [], [], []
            for i in range(10):
                n, m = np.random.randint(len(X)), np.random.randint(len(X))
                X1_new.append(X[n])
                X2_new.append(X[m])
                if Y[n] == Y[m]:
                    Y_new.append(1)
                else:
                    Y_new.append(0)

        X1_new, X2_new, Y_new = torch.tensor(X1_new).to(torch.float32), torch.tensor(X2_new).to(torch.float32), torch.tensor(Y_new).to(torch.float32)
        return X1_new, X2_new, Y_new

    def train_models_repeat(X_train, Y_train):
        if not os.path.exists(PATH_DIR):
            os.makedirs(PATH_DIR)
        def train_models(i, X_train, Y_train):
            model = data_reuploading()
            model.train()

            loss_fn = torch.nn.MSELoss()
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            for it in range(51):

                X1_batch, X2_batch, Y_batch = new_data(X_train, Y_train)
                X1_batch, X2_batch, Y_batch = X1_batch, X2_batch, Y_batch

                pred = model(X1_batch, X2_batch)
                pred, Y_batch = pred.to(torch.float32), Y_batch.to(torch.float32)
                loss = loss_fn(pred, Y_batch)

                opt.zero_grad()
                loss.backward()
                opt.step()

                
                print(f"Iterations: {it} Loss: {loss.item()}")
                PATH = f'{PATH_DIR}/it{i}_{it}.pt'
                torch.save(model.state_dict(), PATH)
        
        for i in range(5):
            train_models(i, X_train, Y_train)

    @qml.qnode(dev, interface="torch")
    def circuit_reuploading(inputs, weights): 
        data_reuploading_embedding(weights, inputs)
        return qml.density_matrix(wires=range(num_qubits))
    

    class Distance_reuploading(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer2 = qml.qnn.TorchLayer(circuit_reuploading, weight_shapes={"weights":2 * num_qubits * num_layers})
        def forward(self, x0, x1):
            rhos1 = self.qlayer2(x1)
            rhos0 = self.qlayer2(x0)

            rho1 = torch.sum(rhos1, dim=0) / len(x1)
            rho0 = torch.sum(rhos0, dim=0) / len(x0)
            rho_diff = rho1 - rho0
            eigvals = torch.linalg.eigvals(rho_diff)
            return 0.5 * torch.real(torch.sum(torch.abs(eigvals)))

    def model_distance(PATH, X0, X1):
        
        model = Distance_reuploading()
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model(X0, X1) 
    
    train_models_repeat(X_train, Y_train)
    PATHs = [[f'{PATH_DIR}/it{i}_{j}.pt' for j in range(51)] for i in range(5)]
    D_train, D_test = [], []
    for PATH in PATHs:
        D_train.append([model_distance(P, X0_train, X1_train).detach().numpy() for P in PATH])
        D_test.append([model_distance(P, X0_test, X1_test).detach().numpy() for P in PATH])
    
    D_train, D_test = np.array(D_train), np.array(D_test)
    D_train_mean, D_test_mean = np.mean(D_train, axis=0), np.mean(D_test, axis=0)
    D_train_std, D_test_std = np.std(D_train, axis=0), np.std(D_test, axis=0)

    np.save(f'nq:{num_qubits}_nl:{num_layers}_nd:{num_data}_D_train_mean.npy', D_train_mean)
    np.save(f'nq:{num_qubits}_nl:{num_layers}_nd:{num_data}_D_test_mean.npy', D_test_mean)
    np.save(f'nq:{num_qubits}_nl:{num_layers}_nd:{num_data}_D_train_std.npy', D_train_std)
    np.save(f'nq:{num_qubits}_nl:{num_layers}_nd:{num_data}_D_test_std.npy', D_test_std)

def run(num_qubits_list, num_data_list, num_layers_list):
    for num_qubits in num_qubits_list:
        for num_layers in num_layers_list:
            for num_data in num_data_list:
                train_embeddings(num_qubits, num_layers, num_data)

#num_qubits_list = [2, 4]
#num_data_list = [1, 5, 10, 100]
#num_layers_list = [1, 5, 10, 20]

num_qubits_list = [4]
num_layers_list = [1,2,3,5,10]
num_data_list = [5, 10, 50]


run(num_qubits_list, num_data_list, num_layers_list)
    