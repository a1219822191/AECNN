import torch.nn as nn
import torch 
import torch.nn.functional as F
import numpy as np
import sys

class NET(nn.Module):
    
    def __init__(self):
        super(NET,self).__init__()
        self.conv1=nn.Conv1d(1,3,20)
        self.ac1=nn.Sigmoid()
        self.pool1=nn.MaxPool1d(2,2)

        self.conv2=nn.Conv1d(3,6,20)

        self.linear=nn.Linear(324,32)
        self.softplus=nn.Softplus()
        self.linear1=nn.Linear(32,1)
        
        
        #self.linear5=nn.Linear(100,64)
        self.conva=nn.Conv1d(1,3,20)
        
    def forward(self, atom_one_hot,atom_env,crystal_atom_idx):
        m0,n0=atom_one_hot.shape
        atom=atom_one_hot.view(m0,1,n0)

        atom=self.conva(atom)
        atom=F.relu(atom)
        s1,s2,s3=atom.shape
        atom=(atom.view(-1,s1*s2*s3).view(s1,s2*s3))
        #atom_one_hot=self.linear5(atom_one_hot)
        


        total=torch.cat((atom,atom_env),1)
        m,n=total.shape
        total=total.reshape(m,1,n)


        one=self.conv1(total)
        one=F.relu(one)
        one=self.pool1(one)

        one=self.conv2(one)
        one=F.relu((one))
        one=self.pool1(one) 

        m1,n1,s1=one.shape
        one=(one.view(-1,m1*n1*s1).view(m1,n1*s1))
        one=self.sum_atom(one,crystal_atom_idx)
        one=F.relu(one)

        one=self.linear(one)
        one=self.softplus(one)
        out=self.linear1(one)
        return out

    def sum_atom(self,atom_imformation,crystal_atom_idx):
        atom_sum=[torch.mean(atom_imformation[idx_map], dim=0,keepdim=True)
            for idx_map in crystal_atom_idx]
        return torch.cat(atom_sum,dim=0)
