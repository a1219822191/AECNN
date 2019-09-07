import torch.nn as nn
import torch 
import torch.nn.functional as F
import numpy as np
import sys

class NET(nn.Module):
    
    def __init__(self):
        super(NET,self).__init__()
        self.conv1=nn.Conv1d(1,3,10)
        self.ac1=nn.Sigmoid()
        #self.bn1=nn.BatchNorm()
        self.pool1=nn.MaxPool1d(2,2)

        self.conv2=nn.Conv1d(3,6,10)

        #self.bn2=nn.BatchNorm1d(1)
        self.bn1=nn.BatchNorm1d(102)
        self.linear=nn.Linear(102,32)
        self.softplus=nn.Softplus()
        self.linear1=nn.Linear(32,1)
        
        self.linear5=nn.Linear(92,64)

    def forward(self, atom_one_hot,atom_env,crystal_atom_idx):
        atom_one_hot=self.linear5(atom_one_hot)
        total=torch.cat((atom_one_hot,atom_env),1)
        m,n=total.shape
        total=total.reshape(m,1,n)


        one=self.conv1(total)
#        print(one.shape)
        #one=self.bn(one)
        one=F.relu(one)
        one=self.pool1(one)

        one=self.conv2(one)
       # print(one.shape)
        one=F.relu((one))
        one=self.pool1(one) 

        m1,n1,s1=one.shape
        #one=self.bn1(one.view(-1,m1*n1*s1).view(m1,n1*s1))
        one=(one.view(-1,m1*n1*s1).view(m1,n1*s1))
        one=self.sum_atom(one,crystal_atom_idx)
        one=F.relu(one)

        one=self.linear(one)
        one=self.softplus(one)
        out=self.linear1(one)
        #one=self.linear1()
        #one=self.batch2(one)
        return out

    def sum_atom(self,atom_imformation,crystal_atom_idx):
        atom_sum=[torch.mean(atom_imformation[idx_map], dim=0,keepdim=True)
            for idx_map in crystal_atom_idx]
        return torch.cat(atom_sum,dim=0)
