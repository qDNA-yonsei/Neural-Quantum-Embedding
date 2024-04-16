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

def train_embeddings(num_qubits, num_data, model_name):
    PATH_DIR = f'nq:{num_qubits}_nd:{num_data}'
    dev = qml.device("default.qubit", wires=num_qubits)
    if model_name == 'PCA-NQE':
        feature_reduction = f'PCA{num_qubits}'
    elif model_name == 'NQE':
        feature_reduction = False
    classes = [0, 1]
    X_, X_t, Y_, Y_t = data.data_load_and_process('mnist', feature_reduction=feature_reduction, classes=classes)
    if feature_reduction == False:
        X_, X_t = torch.tensor(X_).to(torch.float32), torch.tensor(X_t).to(torch.float32)
        X_, X_t = X_.permute(0, 3, 1, 2).detach().numpy(), X_t.permute(0, 3, 1, 2).detach().numpy()


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

    def two_QuantumEmbedding2(x):
        for _ in range(3):
            for i in range(2):
                qml.Hadamard(wires=i)
                qml.RZ(-2 * x[i], wires=i)
            qml.CNOT(wires=[0,1])
            qml.RZ(-2 * x[2], wires=1)
            qml.CNOT(wires=[0,1])
    if num_qubits == 2:
        final_dim = 3
    else:
        final_dim = 2 * num_qubits
    
    @qml.qnode(dev, interface="torch")
    def overlap_NQE(inputs):
        if num_qubits == 2:
            two_QuantumEmbedding2(inputs[0:final_dim])
            qml.adjoint(two_QuantumEmbedding2)(inputs[final_dim:2*final_dim])
        elif num_qubits == 4:
            embedding.Four_QuantumEmbedding2(inputs[0:final_dim])
            qml.adjoint(embedding.Four_QuantumEmbedding2)(inputs[final_dim: 2 * final_dim])
        return qml.probs(wires=range(num_qubits))
    
    class PCA_NQE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer2 = qml.qnn.TorchLayer(overlap_NQE, weight_shapes={})
            self.linear_relu_stack2 = nn.Sequential(
                nn.Linear(num_qubits, 3 * num_qubits),
                nn.ReLU(),
                nn.Linear(3 * num_qubits, 3 * num_qubits),
                nn.ReLU(),
                nn.Linear(3 * num_qubits, final_dim)
            )
        def forward(self, x1, x2):
            x1 = self.linear_relu_stack2(x1)
            x2 = self.linear_relu_stack2(x2)
            x = torch.concat([x1, x2], 1)
            x = self.qlayer2(x)
            return x[:,0]
    
    class NQE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer3 = qml.qnn.TorchLayer(overlap_NQE, weight_shapes={})
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )

            # Layer2: 14 * 14 -> 7 * 7
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )

            # Fully connected Layers 7 * 7 -> 32
            self.fc = nn.Sequential(
                torch.nn.Linear(7 * 7, 32, bias=True),
                torch.nn.Linear(32, final_dim, bias=True))
        
        def forward(self, x1, x2):
            x1 = self.layer1(x1)
            x1 = self.layer2(x1)
            x1 = x1.view(-1, 7 * 7)
            x1 = self.fc(x1)

            x2 = self.layer1(x2)
            x2 = self.layer2(x2)
            x2 = x2.view(-1, 7 * 7)
            x2 = self.fc(x2)

            x = torch.concat([x1, x2], 1)
            x = self.qlayer3(x)
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
            if model_name == 'NQE':
                model = NQE()
            elif model_name == 'PCA-NQE':
                model = PCA_NQE()
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
                PATH = f'{PATH_DIR}/{model_name}_it{i}_{it}.pt'
                torch.save(model.state_dict(), PATH)
        
        for i in range(5):
            train_models(i, X_train, Y_train)
    
    @qml.qnode(dev, interface="torch")
    def circuit_NQE(inputs):
        if num_qubits == 2:
            two_QuantumEmbedding2(inputs)
        elif num_qubits == 4:
            embedding.Four_QuantumEmbedding2(inputs)
        return qml.density_matrix(wires=range(num_qubits))

    
    class Distance_PCA_NQE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_relu_stack2 = nn.Sequential(
                nn.Linear(num_qubits, 3 * num_qubits),
                nn.ReLU(),
                nn.Linear(3 * num_qubits, 3 * num_qubits),
                nn.ReLU(),
                nn.Linear(3 * num_qubits, final_dim)
            )
            self.qlayer2 = qml.qnn.TorchLayer(circuit_NQE, weight_shapes={})
        def forward(self, x0, x1):
            x1 = self.linear_relu_stack2(x1)
            x0 = self.linear_relu_stack2(x0)

            rhos1 = self.qlayer2(x1)
            rhos0 = self.qlayer2(x0)
            rho1 = torch.sum(rhos1, dim=0) / len(x1)
            rho0 = torch.sum(rhos0, dim=0) / len(x0)
            rho_diff = rho1 - rho0
            eigvals = torch.linalg.eigvals(rho_diff)
            return 0.5 * torch.real(torch.sum(torch.abs(eigvals)))
        
    class Distance_NQE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

            # Layer2: 14 * 14 -> 7 * 7
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )

            # Fully connected Layers 7 * 7 -> 7
            self.fc = nn.Sequential(
                torch.nn.Linear(7 * 7, 32, bias=True),
                torch.nn.Linear(32, final_dim, bias=True))
            self.qlayer2 = qml.qnn.TorchLayer(circuit_NQE, weight_shapes={})
        def forward(self, x0, x1):
            
            x1 = self.layer1(x1)
            x1 = self.layer2(x1)
            x1 = x1.view(-1, 7 * 7)
            x1 = self.fc(x1)

            x0 = self.layer1(x0)
            x0 = self.layer2(x0)
            x0 = x0.view(-1, 7 * 7)
            x0 = self.fc(x0)

            rhos1 = self.qlayer2(x1)
            rhos0 = self.qlayer2(x0)
            rho1 = torch.sum(rhos1, dim=0) / len(x1)
            rho0 = torch.sum(rhos0, dim=0) / len(x0)
            rho_diff = rho1 - rho0
            eigvals = torch.linalg.eigvals(rho_diff)
            return 0.5 * torch.real(torch.sum(torch.abs(eigvals)))
    

    def model_distance(PATH, X0, X1):
        if model_name == 'PCA-NQE':
            model = Distance_PCA_NQE()
        else:
            model = Distance_NQE()
        
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model(X0, X1) 
    
    train_models_repeat(X_train, Y_train)
    PATHs = [[f'{PATH_DIR}/{model_name}_it{i}_{j}.pt' for j in range(51)] for i in range(5)]
    D_train, D_test,  = [], []
    for PATH in PATHs:
        D_train.append([model_distance(P, X0_train, X1_train).detach().numpy() for P in PATH])
        D_test.append([model_distance(P, X0_test, X1_test).detach().numpy() for P in PATH])

    
    D_train, D_test = np.array(D_train), np.array(D_test)
    D_train_mean, D_test_mean = np.mean(D_train, axis=0), np.mean(D_test, axis=0)
    D_train_std, D_test_std = np.std(D_train, axis=0), np.std(D_test, axis=0)

    np.save(f'{model_name}_nq:{num_qubits}_nd:{num_data}_D_train_mean.npy', D_train_mean)
    np.save(f'{model_name}_nq:{num_qubits}_nd:{num_data}_D_test_mean.npy', D_test_mean)
    np.save(f'{model_name}_nq:{num_qubits}_nd:{num_data}_D_train_std.npy', D_train_std)
    np.save(f'{model_name}_nq:{num_qubits}_nd:{num_data}_D_test_std.npy', D_test_std)

def run(num_qubits_list, num_data_list, model_name_list):
    for num_qubits in num_qubits_list:
        for num_data in num_data_list:
            for model_name in model_name_list:
                train_embeddings(num_qubits, num_data, model_name)

num_qubits_list = [2,4]
num_data_list = [5, 10, 50]
model_name_list = ['NQE']

run(num_qubits_list, num_data_list, model_name_list)
    