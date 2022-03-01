import sys, os
sys.path.append(os.environ['d'])
from utils.opt import *
from build_lm import *
from dataloader_lm import *
from torch.optim import *
gpu0 = torch.device('cpu')
if torch.cuda.is_available():
    gpu0 = torch.device('cuda:0')


tprint('buiding language model...')
model = build_lm()
model.load_state_dict(torch.load(config.lm_model_out))
model = model.to(gpu0)


tprint('building language dataloader...')
data = dataloader_lm()

tprint('buiding optim, criterion...')
# opt = SGD(model.parameters(), momentum = 0.9, lr = config.pre_lr)
#　sch = lr_scheduler.OneCycleLR(opt, max_lr=config.pre_lr * 50, total_steps = 8000)
opt = Ranger(model.parameters(), lr = config.pre_lr)
opt.zero_grad()
criterion = nn.CrossEntropyLoss()



losses = []
accs = []
running_step = 1
while True:
    inputs, labels = data.read()
    inputs = inputs.to(gpu0)
    labels = labels.to(gpu0)
    # forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    if running_step * config.lm_batch % config.lm_step_samples == 0:
        opt.step()
        #　sch.step()
        opt.zero_grad()
        
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    losses.append(loss.item())
    accs.append(correct / len(labels))
    if running_step * config.lm_batch % config.lm_show_samples == 0:
        loss = sum(losses) / len(losses)
        acc = sum(accs) / len(accs)
        losses = []
        accs = []
        with open(config.lm_record_out, 'a+', encoding = 'utf-8') as fout:
            fout.write(str(running_step * config.lm_batch) + ' ' + str(loss) + ' ' + str(acc) + '\n')
        print('[%8d] loss: %.3f acc: %.3f' %
              (running_step * config.lm_batch, loss, acc))
    if running_step * config.lm_batch % config.lm_save_samples == 0:
        torch.save(model.state_dict(), config.lm_model_out)
    running_step += 1
