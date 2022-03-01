import sys, os

sys.path.append(os.environ['d'])
import torch.nn.functional

from utils.opt import *
from build_access import *
from dataloader_access import *
from torch.optim import *
import numpy as np

gpu0 = torch.device('cpu')
if torch.cuda.is_available():
    gpu0 = torch.device('cuda:0')

tprint('buiding access model...')
model = build_access()
model.load_state_dict(torch.load(config.access_model_out))
model = model.to(gpu0)

tprint('building access dataloader...')
data_train = dataloader_access(config.access_data_name, 'train', None)
data_valid = dataloader_access(config.access_data_name, 'valid', None)


tprint('buiding optim, criterion...')
# opt = SGD(model.parameters(), momentum = 0.9, lr = config.pre_lr)
# 　sch = lr_scheduler.OneCycleLR(opt, max_lr=config.pre_lr * 50, total_steps = 8000)
opt = Ranger(model.parameters(), lr=config.access_lr)
opt.zero_grad()
criterion = nn.BCEWithLogitsLoss()


for epc in range(config.access_epochs):
    step = 0
    losses = []
    accs = []
    for inputs, labels in data_train:
        inputs = inputs.to(gpu0)
        labels = labels.to(gpu0)
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1).float())
        loss.backward()
        opt.step()
        opt.zero_grad()

        correct = ((torch.sigmoid(outputs.data).view(-1)>0.5)==labels).sum().item()
        accs.append(correct / config.access_batch)
        losses.append(loss.item())
        if step % 100 == 0:
            print('[%4d %5d %5d] loss: %.3f acc: %.3f' %
                  (epc, step, len(data_train), np.mean(losses), np.mean(accs)))
            accs = []
            losses = []
        step += 1
    model.eval()
    torch.save(model.state_dict(), config.access_model_out)
    val_losses = []
    val_accs = []
    with torch.no_grad():
        for inputs, labels in data_valid:
            inputs = inputs.to(gpu0)
            labels = labels.to(gpu0)
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1).float())
            correct = ((torch.sigmoid(outputs.data).view(-1)>0.5)==labels).sum().item()
            val_losses.append(loss.item())
            val_accs.append(correct / config.access_batch)
    print('[%4d valid] loss: %.3f acc: %.3f' %
          (epc, np.mean(val_losses), np.mean(val_accs)))
    model.train()