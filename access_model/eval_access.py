import sys, os

# sys.path.append(os.environ['d'])
import torch.nn.functional

from utils.opt import *
from build_access import *
from dataloader_access import *
from torch.optim import *
import numpy as np
from sklearn import metrics

gpu0 = torch.device('cpu')
if torch.cuda.is_available():
    gpu0 = torch.device('cuda:0')

tprint('buiding access model...')
model = build_access()
model.load_state_dict(torch.load(config.access_model_out))
model = model.to(gpu0)

tprint('building access dataloader...')
data_test = dataloader_access(config.access_data_name, 'test', None)


tprint('buiding criterion...')
# opt = SGD(model.parameters(), momentum = 0.9, lr = config.pre_lr)
# 　sch = lr_scheduler.OneCycleLR(opt, max_lr=config.pre_lr * 50, total_steps = 8000)

criterion = nn.BCEWithLogitsLoss()



fout = []
flabel = []
with torch.no_grad():
    for inputs, labels in data_test:
        inputs = inputs.to(gpu0)
        labels = labels.to(gpu0)
        # forward + backward + optimize
        outputs = model(inputs)
        fout.extend(torch.sigmoid(outputs.data).view(-1).tolist())
        flabel.extend(labels.view(-1).tolist())
fpred = [1 if i > 0.5 else 0 for i in fout]
print('Calculating AUC...')
# print(pred_y, test_y)

auroc = metrics.roc_auc_score(flabel, fout)
auprc = metrics.average_precision_score(flabel, fout)
f1 = metrics.f1_score(flabel, fpred)
print(f1, auroc, auprc)