#firom __future__ import print_function, division
import os
import csv
import re
import json
import functools
import random
import warnings
import math
from ase.io import read 
import sys

import torch
import numpy as np
from soaplite import getBasisFunc, get_periodic_soap_locals ,get_periodic_soap_structure,get_periodic_soap_structure_gauss
import quippy
from quippy import descriptors
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.core.structure import Structure
from libwacsf.wacsf import WACSF
from ase.io import read
def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_size=None,
                              val_size=1000, test_size=1000, return_test=False,
                              num_workers=1, pin_memory=False):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    batch_size: int
    train_size: int
    val_size: int
    test_size: int
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if train_size is None:
        assert val_size + test_size < total_size
        print('[Warning] train_size is None, using all training data.')
    else:
        assert train_size + val_size + test_size <= total_size
    indices = list(range(total_size))
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
                    indices[-(val_size+test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn,
                               pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                  pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.
    """
    #print(len(dataset_list))
    batch_atom_fea=[]
    batch_soap=[]
    batch_target=[]
    crystal_atom_idx=[]
    batch_cif_ids = []
    base_idx=0

    for i,((atom_fea,struc),target,cif_id) in enumerate(dataset_list):
        n_a=atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_soap.append(struc)
        new_idx = torch.LongTensor(np.arange(n_a)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        base_idx += n_a
        batch_cif_ids.append(cif_id)
    return (torch.cat(batch_atom_fea,dim=0),
            torch.cat(batch_soap,dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids

#Code element-------------------------------------------------------------             
#class AtomInitializer(object):
#    def __init__(self, atom_types):
#        self.atom_types = set(atom_types)
#        self._embedding = {}
#
#
#    def get_atom_fea(self, atom_type):
#        assert atom_type in self.atom_types
#        return self._embedding[atom_type]
#
#    def load_state_dict(self, state_dict):
#        self._embedding = state_dict
#        self.atom_types = set(self._embedding.keys())
#        self._decodedict = {idx: atom_type for atom_type, idx in
#                                self._embedding.items()}
#
#    def state_dict(self):
#        return self._embedding
#
#    def decode(self, idx):
#        if not hasattr(self, '_decodedict'):
#            self._decodedict = {idx: atom_type for atom_type, idx in
#                                    self._embedding.items()}
#        return self._decodedict[idx]
#
#
#class AtomCustomJSONInitializer(AtomInitializer):
#
#    def __init__(self, elem_embedding_file):
#        with open(elem_embedding_file) as f:
#            elem_embedding = json.load(f)
#        elem_embedding = {int(key): value for key, value
#                            in elem_embedding.items()}
#
#        atom_types = set(elem_embedding.keys())
#        super(AtomCustomJSONInitializer, self).__init__(atom_types)
#        for key, value in elem_embedding.items():
#            self._embedding[key] = np.array(value, dtype=float)

#class one_hot_element(object)
#
#    def __init__(self,element):
#
#
#       # ChemicalSymbols = [ 'X',  'H',  'He', 'Li', 'Be','B',  'C',  'N',  'O',  'F',
#       #             'Ne', 'Na', 'Mg', 'Al', 'Si','P',  'S',  'Cl', 'Ar', 'K',
#       #             'Ca', 'Sc', 'Ti', 'V',  'Cr','Mn', 'Fe', 'Co', 'Ni', 'Cu',
#       #             'Zn', 'Ga', 'Ge', 'As', 'Se','Br', 'Kr', 'Rb', 'Sr', 'Y',
#       #             'Zr', 'Nb', 'Mo', 'Tc', 'Ru','Rh', 'Pd', 'Ag', 'Cd', 'In',
#       #             'Sn', 'Sb', 'Te', 'I',  'Xe','Cs', 'Ba', 'La', 'Ce', 'Pr',
#       #             'Nd', 'Pm', 'Sm', 'Eu', 'Gd','Tb', 'Dy', 'Ho', 'Er', 'Tm',
#       #             'Yb', 'Lu', 'Hf', 'Ta', 'W','Re', 'Os', 'Ir', 'Pt', 'Au',
#       #             'Hg', 'Tl', 'Pb', 'Bi', 'Po','At', 'Rn', 'Fr', 'Ra', 'Ac',
#       #             'Th', 'Pa', 'U',  'Np', 'Pu','Am', 'Cm', 'Bk', 'Cf', 'Es',
#       #             'Fm', 'Md', 'No', 'Lr']        
def one_hot_element(self,ele):
    self.one_hot=[0, 0, 0, 0, 0, 0, 0, 0, 0,  
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0] 

    self.atomicNum = {'X': 0, 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 
                    'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 
                    'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 
                    'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 
                    'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 
                    'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 
                    'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 
                    'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 
                    'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 
                    'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 
                    'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 
                    'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 
                    'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103}
    self.one_hot(ele)=1

    return self.one_hot

        
#---------------------------------------------------------------------------


class CIFData(Dataset): 
        # read file
    def __init__ (self,root_dir,random_seed=123):
        self.root_dir = root_dir
        struc =[]
        bb = []
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)

    # data num    
    def __len__(self):
        return len(self.id_prop_data)
    # input_file target
    @functools.lru_cache(maxsize=None)
    def __getitem__(self,idx):
        cif_id, target = self.id_prop_data[idx]
        
        crystal = Structure.from_file(os.path.join(self.root_dir,cif_id+'.cif'))
        fi=read(os.path.join(self.root_dir,cif_id+'.cif'))
        pos=fi.positions
        lat=fi.cell
         #---Code structure----------------------
        a = WACSF(rcut=6.0,nfeature= 33)
        struc =a.car2wacsf(lat,pos)
        #-------------------------------------------
        # Code element
        ele=[]
        for i in range(len(crystal)):
            ele.append(crystal[i].specie.number)
        #ele1=sorted(ele)
        atom_fea = np.vstack([self.ont_hot_elemen(ele[i])
                                for i in range(len(crystal))])
        #--------------------------------------------
        target = torch.FloatTensor([float(target)]) 
        struc = torch.Tensor(struc)
        atom_fea = torch.Tensor(atom_fea)
        return (atom_fea,struc) , target,cif_id
