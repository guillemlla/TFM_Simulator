from abc import ABC
import numpy as np
np.random.seed(123)
from NN_Models import Adapter_Loss_Funcion
import torch
import torch.nn as nn
import torch.optim as optim
import logging

PATH = "data/"
FILENAME_PPS = "adapter_pps.model.pt"
FILENAME_QUEUE = "adapter_queue.model.pt"


class Adapter(object):

    def __init__(self, num_slices):
        self.hparams = {'batch_size': 1, 'num_epochs': 5, 'test_batch_size': 1, 'hidden_size': 128,
                        'num_classes': num_slices, 'num_inputs': num_slices, 'learning_rate': 1e-3,
                        'log_interval': 1,
                        'device': 'cuda' if torch.cuda.is_available() else 'cpu'}

        self.network_queue = AdapterModel(hparams=self.hparams)
        self.network_pps = AdapterModel(hparams=self.hparams)

        self.optimizer_queue = optim.RMSprop(self.network_queue.parameters(), lr=self.hparams['learning_rate'])
        self.optimizer_pps = optim.RMSprop(self.network_pps.parameters(), lr=self.hparams['learning_rate'])
        self.criterion = Adapter_Loss_Funcion.loss_function

        self.train_losses = []
        self.test_losses = []
        self.test_accs = []

    def correct_predictions(self, predicted_batch, label_batch):
        accum_error = sum(abs(predicted_batch - label_batch)) / len(predicted_batch)
        return torch.sum(accum_error) / self.hparams['num_classes']

    def train_epoch(self, train_loader, epoch):
        self.network_queue.train()
        self.network_pps.train()
        device = self.hparams['device']
        logging.debug("Loading batch")
        for batch_idx, (data, target) in enumerate(train_loader):
            logging.debug("Batch loaded")
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            self.optimizer_queue.zero_grad()
            self.optimizer_pps.zero_grad()
            output_queue = self.network_queue(data)
            output_pps = self.network_pps(data)
            loss_queue, loss_pps = self.criterion(output_queue, output_pps, target)
            loss_queue.backward()
            loss_pps.backward()
            avg_loss_queue = loss_queue.item()
            avg_loss_pps = loss_pps.item()
            self.optimizer_pps.step()
            self.optimizer_queue.step()
            if batch_idx % self.hparams['log_interval'] == 0:
                logging.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss queue: {:.6f} Loss PPS: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss_queue.item(), loss_pps.item()))
        return avg_loss_queue, avg_loss_pps

    def test_epoch(self, test_loader):
        self.network_queue.eval()
        self.network_pps.eval()
        device = self.hparams['device']
        test_loss_queue = 0
        test_loss_pps = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.type(torch.FloatTensor)
                data = data.to(device)
                self.optimizer_pps.zero_grad()
                self.optimizer_queue.zero_grad()
                output_queue = self.network_queue(data)
                output_pps = self.network_pps(data)
                loss_queue, loss_pps = self.criterion(output_queue, output_pps, target)
                test_loss_pps += loss_pps
                test_loss_queue += loss_queue
        test_loss_queue /= len(test_loader.dataset)
        test_loss_pps /= len(test_loader.dataset)
        logging.debug('\nTest set: Average loss Queue: {:.4f}, Average loss PPS: {:.4f})\n'.format(
            test_loss_queue, test_loss_pps))
        return test_loss_queue, test_loss_pps

    def save_model(self):
        torch.save(self.network_pps.state_dict(), PATH + FILENAME_PPS)
        torch.save(self.network_queue.state_dict(), PATH + FILENAME_QUEUE)

    def load_model(self):
        self.network_pps.load_state_dict(torch.load(PATH + FILENAME_PPS))
        self.network_queue.load_state_dict(torch.load(PATH + FILENAME_QUEUE))


class AdapterModel(nn.Module, ABC):
    def __init__(self, hparams):
        super(AdapterModel, self).__init__()
        self.lin1 = nn.Linear(hparams['num_inputs'], hparams['hidden_size'])
        self.Relu1 = nn.ReLU()
        self.lin2 = nn.Linear(hparams['hidden_size'], hparams['hidden_size'])
        self.Relu2 = nn.ReLU()
        self.lin3 = nn.Linear(hparams['hidden_size'], hparams['num_classes'])
        self.Act3 = nn.Softmax()

    def forward(self, x):
        x = self.lin1(x)
        x = self.Relu1(x)
        x = self.lin2(x)
        x = self.Relu2(x)
        x = self.lin3(x)
        x = self.Act3(x)
        return x
