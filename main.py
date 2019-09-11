from __future__ import print_function                                        
import argparse
import time
import shutil  
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR 
#import torchvision.transforms as transforms
#import torchvision.utils as vutils
import torch.optim as optim
from torch.autograd import Variable
import os
import sys 
import numpy as np
	
from random import sample
from lae.data import CIFData
from lae.data import collate_pool, get_train_val_test_loader
#from lae.model1 import SAOP_atom
from lae.model import NET
from torch.utils.data import DataLoader
parser = argparse.ArgumentParser(description='AECNN')

parser.add_argument('--dataroot',metavar='OPTIONS', nargs='+',default=None,help='path to dataset')
parser.add_argument('--disable-cuda', action='store_true',help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,metavar='LR', help='initial learning rate (default: ''0.01)')
parser.add_argument('--lr-milestones', default=[200], nargs='+', type=int,metavar='N', help='milestones for scheduler (default:'
                    '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--train-size', default=16810, type=int, metavar='N',
                    help='number of training data to be loaded (default none)')
parser.add_argument('--val-size', default=1000, type=int, metavar='N',
                    help='number of validation data to be loaded (default '
                                        '1000')
parser.add_argument('--test-size', default=1000, type=int, metavar='N',
                    help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
#parser.add_argument('--atom_fea_len', default=64, type=int, metavar='N',
#                    help='number of hidden atom features in conv layers')
#parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
#                    help='number of hidden features after pooling')
#parser.add_argument('--n-conv', default=3, type=int, metavar='N',
#                    help='number of conv layers')
#parser.add_argument('--n-h', default=1, type=int, metavar='N',
#                    help='number of hidden layers after pooling')

opt = parser.parse_args(sys.argv[1:])
opt.cuda = not opt.disable_cuda and torch.cuda.is_available()
best_mae_error = 1e10
def main():
    global opt, best_mae_error

    dataset = CIFData(*opt.dataroot)
    collate_fn = collate_pool
    
    train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,collate_fn=collate_fn,batch_size=opt.batch_size,
            train_size=opt.train_size, num_workers=opt.workers,
            val_size=opt.val_size, test_size=opt.test_size,pin_memory=opt.cuda,
            return_test=True)
    # obtain target value normalizer
    sample_data_list = [dataset[i] for i in
                        sample(range(len(dataset)), 1000)]
    input, sample_target,_ = collate_pool(sample_data_list)
    input_1=input[0]
    normalizer = Normalizer(sample_target)
    s = Normalizer(input_1)


    model=NET()

    if torch.cuda.is_available():
        print('cuda is ok')
        model = model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), opt.lr,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))   

        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    scheduler = MultiStepLR(optimizer, milestones=opt.lr_milestones,
                            gamma=0.1)
    for epoch in range(opt.start_epoch,opt.epochs):
        train(train_loader, model, criterion, optimizer, epoch,normalizer,s)

        mae_error = validate(val_loader, model, criterion, normalizer,s) 

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        
        save_checkpoint({ 
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'opt': vars(opt)
        }, is_best)
        # test bset model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, normalizer, s,test=True)
def train(train_loader, model, criterion, optimizer, epoch,normalizer,s):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    model.train()

    end = time.time()
#    with torchsnooper.snoop():
    for i, (input,target,_) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if opt.cuda:
        #    input[0]=s.norm(input[0])
            input_var=(Variable(input[0].cuda()),
                Variable(input[1].cuda()),
                [crys_idx.cuda() for crys_idx in input[2]])
        else:
            input_var=(Variable(input[0]),
                    Variable(input[1]),
                    input[3])
        
        target_normed = normalizer.norm(target)
        if opt.cuda:
            target_var = Variable(target_normed).cuda()
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

         # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                    )

def validate(val_loader, model, criterion, normalizer,s,test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    model.eval()

    end = time.time()

    for i, (input, target,batch_cif_ids) in enumerate(val_loader):
        if opt.cuda:
            
            with torch.no_grad():
    #            input[0]=s.norm(input[0])
                input_var=(Variable(input[0].cuda()),
                        Variable(input[1].cuda()), 
                        [crys_idx.cuda() for crys_idx in input[2]])
        else:
            input_var=(Variable(input[0]),
                    Variable(input[1]),
                    input[3])
        
        target_normed = normalizer.norm(target)

        if opt.cuda:
            with torch.no_grad():

                target_var = Variable(target_normed.cuda())
        else:
            target_var = Variable(target_normed,volatile=True)


        output = model(*input_var)
        loss = criterion(output, target_var)

        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                mae_errors=mae_errors))

    if test:
            star_label = '**'
            import csv
            with open('test_results.csv', 'w') as f:
                writer = csv.writer(f)
                for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                                test_preds):
                    writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                            mae_errors=mae_errors))
    return mae_errors.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)
    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
def mae(prediction, target):
    #Computes the mean absolute error between prediction and target
    return torch.mean(torch.abs(target - prediction))
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
if __name__ =='__main__':
    main()
