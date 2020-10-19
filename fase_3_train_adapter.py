import torch
import pickle
import matplotlib.pyplot as plt
from NN_Models.Adapter import Adapter
import logging
from datetime import datetime

dateTime = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
logging.basicConfig(filename='logs/log_fase_3_' + dateTime + '.log', level=logging.DEBUG)


with open('adapter.dataset', 'rb') as config_dictionary_file:
    predictor_dataset = pickle.load(config_dictionary_file)

dataset = predictor_dataset['dataset']
num_slices = predictor_dataset['num_slices']


adapter = Adapter(num_slices)

total = len(dataset)

train_losses_pps = []
train_losses_queue = []
test_losses_queue = []
test_losses_pps = []


train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=adapter.hparams['batch_size'],
    shuffle=False)

test_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=adapter.hparams['batch_size'],
    shuffle=False)

for epoch in range(1, adapter.hparams['num_epochs'] + 1):
    avg_loss_queue, avg_loss_pps = adapter.train_epoch(train_loader, epoch)
    train_losses_pps.append(avg_loss_pps)
    train_losses_queue.append(avg_loss_queue)
    test_loss_queue, test_loss_pps = adapter.test_epoch(test_loader)
    test_losses_pps.append(test_loss_pps)
    test_losses_queue.append(test_loss_queue)

logging.debug("SAVING ADAPTER MODEL")
adapter.save_model()
logging.debug("ADAPTER MODEL SAVED")


def plot(train_losses, test_losses, test_accs):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.xlabel('Epoch')
    plt.ylabel('mse_loss')
    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='test')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Error')
    plt.plot(test_accs)

    plt.show()
