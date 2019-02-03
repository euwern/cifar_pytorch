import torch
import torch.nn as nn
import torchvision.models as models
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import utils
from tqdm import tqdm
import os
import argparse
from torchvision import datasets
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument('--is_test', action='store_true', default=False)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--dataset', type=str, default='cifar10_4000', help='dataset')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--model', choices=['base', 'pretrain'], default='base')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--lr_steps', default=[80], nargs='+', type=int)
parser.add_argument('--dtype', choices=['pickle', 'image', 'hdf5'], default='pickle')

args = parser.parse_args()
save_path = 'results/%s' % (args.dataset)
save_path = save_path + '/' + args.model

use_pretrain = 'pretrain' in args.model
if not os.path.exists(save_path):
    os.makedirs(save_path)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

dataset = utils.__dict__['ImageDataset_' + args.dtype]

def save_checkpoint():
    checkpoint = [model.state_dict(), opt.state_dict()]
    torch.save(checkpoint, '%s/checkpoint_%d_%d.pth' % (save_path, args.seed, epoch))
    
def load_checkpoint(load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint[0])
    opt.load_state_dict(checkpoint[1])

def compute_acc(class_out, targets):
    preds = torch.max(class_out, 1)[1]
    softmax  = torch.exp(class_out[0])
    pos = 0; 
    for ix in range(preds.size(0)):
        if preds[ix] == targets[ix]:
            pos = pos + 1
    accuracy = pos / preds.size(0) * 100

    return accuracy

def train():
    model.train()
    avg_loss = 0
    avg_acc = 0
    avg_real_acc = 0
    avg_fake_acc = 0
    count = 0
    for _, (data, target) in enumerate(tqdm(data_loader)):
        opt.zero_grad()
        data, target  = Variable(data).cuda(), Variable(target.long()).cuda()
        out = model(data)
        loss = ent_loss(out, target)
        loss.backward()
        opt.step()
        avg_loss = avg_loss + loss.item()
        curr_acc = compute_acc(out.data, target.data)
        avg_acc = avg_acc + curr_acc
        count = count + 1
    avg_loss = avg_loss / count
    avg_acc = avg_acc / count
    print('Epoch: %d; Loss: %f; Acc: %.2f; ' % (epoch, avg_loss, avg_acc))
    loss_logger.log(str(avg_loss))
    acc_logger.log(str(avg_acc))
    return avg_loss

def test():
    load_checkpoint('%s/checkpoint_%d_%d.pth' % (save_path, args.seed, args.max_epochs))
    model.eval()
    data_loader = torch.utils.data.DataLoader(dataset(dataset=args.dataset + '_' + str(args.seed), train=False, use_pretrain=use_pretrain), batch_size=args.batch_size, num_workers=4)
   
    pos=0; total=0;
    prediction_list = []
    groundtruth_list = []
    for _, (data, target) in enumerate(tqdm(data_loader)):
        data, target  = Variable(data).cuda(), Variable(target.long()).cuda()
        out = model(data)
        pred = torch.max(out, out.dim() - 1)[1]
        pos = pos + torch.eq(pred.cpu().long(), target.data.cpu().long()).sum().item()
        total = total + data.size(0)
    acc = pos * 1.0 / total * 100
    print(acc)
    print('Acc: %.2f' % acc)

args.num_class = None
if args.dataset.startswith('cifar10_'):
    args.num_class = 10
elif args.dataset.startswith('cifar100_'):
    args.num_class = 100

if not args.is_test:
    data_loader = torch.utils.data.DataLoader(dataset(dataset=args.dataset + '_' + str(args.seed), train=True, use_pretrain=use_pretrain), batch_size=args.batch_size, shuffle=True, num_workers=4)


class Model(nn.Module):
    def __init__(self, num_class, use_pretrain):
        super(Model, self).__init__()

        base = models.__dict__['resnet18'](pretrained=use_pretrain)
        self.base = nn.Sequential(*list(base.children())[:-1])
        self.fc1 = nn.Linear(512, num_class)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.base(input).squeeze()
        output = self.relu(output)
        output = self.fc1(output)

        return output

model = Model(args.num_class, use_pretrain)
model = model.cuda()

if args.model == 'pretrain':
    opt = optim.SGD([{'params':model.base.parameters(), 'lr': args.lr * 1e-3}, 
                     {'params':model.fc1.parameters()}], lr=args.lr, momentum=0.9, nesterov=False, weight_decay=5e-4)
else:
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False, weight_decay=5e-4)
                
sch = lr_scheduler.MultiStepLR(opt, milestones=args.lr_steps, gamma=0.1) #80 -- 0.1 no nestrov


if not os.path.exists(save_path):
    os.makedirs(save_path)

if not args.is_test:
    loss_logger = utils.TextLogger('loss', '{}/loss_{}.log'.format(save_path, args.seed))
    acc_logger = utils.TextLogger('acc', '{}/acc_{}.log'.format(save_path, args.seed))

ent_loss = nn.CrossEntropyLoss().cuda()

epoch = 1
if not args.is_test:
    while True:
        loss = train()
        if epoch == args.max_epochs:
            save_checkpoint()
        if epoch == args.max_epochs:
            break
        print(opt.param_groups[0]['lr'])   
        sch.step(epoch)
        epoch = epoch + 1
    test()
else:
    test()
