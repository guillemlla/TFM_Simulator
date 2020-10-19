from NN_Models.Predictor import Predictor
import torch
import matplotlib.pyplot as plt
import pickle
import logging
from datetime import datetime
import matplotlib.font_manager as font_manager

dateTime = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
logging.basicConfig(filename='logs/log_fase_2_' + dateTime + '.log', level=logging.DEBUG)

with open('predictor.dataset', 'rb') as config_dictionary_file:
    predictor_dataset = pickle.load(config_dictionary_file)

dataset = predictor_dataset['dataset']
num_slices = predictor_dataset['num_slices']

predictor = Predictor(num_slices)

total = len(dataset)

train_losses = []
val_losses = []

num_train = int(len(dataset) * 0.70)
num_val = int(len(dataset) * 0.15)
num_test = len(dataset) - int(len(dataset) * 0.85)

random_split = True

if random_split:
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [num_train, num_val, num_test])
else:
    train_set = [dataset.__getitem__(i) for i in range(0, num_train)]
    val_set = [dataset.__getitem__(i) for i in range(num_train, num_train + num_val)]
    test_set = [dataset.__getitem__(i) for i in range(num_train + num_val, len(dataset))]


train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=predictor.hparams['batch_size'],
    shuffle=False)

validation_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=predictor.hparams['batch_size'],
    shuffle=False)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=predictor.hparams['batch_size'],
    shuffle=False)

for epoch in range(1, predictor.hparams['num_epochs'] + 1):
    train_losses.append(predictor.train_epoch(train_loader, epoch))
    val_loss = predictor.test_epoch(validation_loader)
    val_losses.append(val_loss)


test_loss = predictor.test_epoch(test_loader)

plt.figure(figsize=(20, 16))
fig, ax = plt.subplots()
ax.tick_params(axis='both', which='major', labelsize=10000)
ax.tick_params(axis='both', which='minor', labelsize=20000)

plt.subplot(1, 1, 1)
plt.xlabel('Epoch', fontsize=15, fontweight='bold')
plt.ylabel('MSE loss', fontsize=15, fontweight='bold')
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.plot(len(val_losses) - 1, test_loss, 'ro',  label='Test Loss')
font = font_manager.FontProperties(style='normal', size=10)
plt.legend(prop=font)


plt.show()



predictor.save_model()

logging.debug('Model Saved')
