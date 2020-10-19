from abc import ABC
import numpy as np

np.random.seed(123)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

PATH = "data/predictor.model.pt"


class Predictor(object):

    def __init__(self, num_slices):
        self.hparams = {'batch_size': 10, 'num_epochs': 40, 'test_batch_size': 64, 'hidden_size': 128,
                        'num_classes': num_slices, 'num_inputs': num_slices + 2, 'learning_rate': 1e-3,
                        'log_interval': 1,
                        'device': 'cuda' if torch.cuda.is_available() else 'cpu'}

        self.network = PredictorModel(self.hparams)

        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.hparams['learning_rate'])
        self.criterion = F.mse_loss

        self.train_losses = []
        self.test_losses = []
        self.test_accs = []

    def correct_predictions(self, predicted_batch, label_batch):
        accum_error = sum(abs(predicted_batch - label_batch)) / len(predicted_batch)
        return torch.sum(accum_error) / self.hparams['num_classes']

    def train_epoch(self, train_loader, epoch):
        self.network.train()
        device = self.hparams['device']
        avg_loss = None
        avg_weight = 0.1
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor)
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.network(data)
            logging.debug('OUTPUT' + str(output))
            logging.debug('TARGET' + str(target))
            loss = self.criterion(output, target)
            loss.backward()
            if avg_loss:
                avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
            else:
                avg_loss = loss.item()
            self.optimizer.step()
            if batch_idx % self.hparams['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
        return avg_loss

    def test_epoch(self, test_loader):
        self.network.eval()
        device = self.hparams['device']
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.type(torch.FloatTensor)
                target = target.type(torch.FloatTensor)
                data, target = data.to(device), target.to(device)
                output = self.network(data)
                test_loss += self.criterion(output, target, reduction='sum').item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}\n'.format(
            test_loss
        ))
        return test_loss

    def save_model(self):
        torch.save(self.network.state_dict(), PATH)

    def load_model(self):
        self.network.load_state_dict(torch.load(PATH))


class PredictorModel(nn.Module, ABC):
    def __init__(self, hparams):
        super(PredictorModel, self).__init__()
        self.lin1 = nn.Linear(hparams['num_inputs'], hparams['hidden_size'])
        self.Relu1 = nn.ReLU()
        self.lin2 = nn.Linear(hparams['hidden_size'], hparams['hidden_size'])
        self.Relu2 = nn.ReLU()
        self.lin3 = nn.Linear(hparams['hidden_size'], hparams['num_classes'])
        self.Softmax = nn.Softmax()

    def forward(self, x):
        x = self.lin1(x)
        x = self.Relu1(x)
        x = self.lin2(x)
        x = self.Relu2(x)
        x = self.lin3(x)
        x = self.Softmax(x)
        return x


class RNN(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.rnn = nn.RNN(input_size=hparams['num_inputs'], hidden_size=hparams['hidden_size'], num_layers=2,
                          nonlinearity='relu')
        self.relu = nn.ReLU()
        self.lin = nn.Linear(hparams['hidden_size'], hparams['num_classes'])
        self.softmax = nn.Softmax()

    def forward(self, x):
        x, hn = self.rnn(x.view([x.shape[0], 1, 6]))
        x = self.relu(x)
        x = self.lin(x)
        x = self.softmax(x)
        return x
