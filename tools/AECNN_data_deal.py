import argparse
import glob
import argparse
from ase.io import read
from libwacsf.wacsf import WACSF
from libgap.WACSF import Wacsf
import h5py
from pymatgen.core.structure import Structure
import numpy as np
import torch
import sys
import os
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default=None, help='Where to store structures file *.cif')
opt = parser.parse_args()
class structrue_analy(object):
    def __init__(self):
        pass

    def analy(self,path):
        xyzfile = glob.glob(path + "/*.cif")
        for xyz in xyzfile:
            cif_id=xyz.split('/')[-1]
            cif_id=cif_id.split('.')[0]
            crystal = Structure.from_file(xyz)
            fi=read(xyz)
            pos=fi.positions
            lat=fi.cell
            #species=fi.get_atomic_numbers()
            #wacsf = Wacsf(nf = 66, rcut = 6.0, lgrad = False)
            #struc = wacsf.car2wacsf(lat, species, pos)

            a = WACSF(rcut=8.0,nfeature= 50)
            struc =a.car2wacsf(lat,pos)
            ele=[]
            for i in range(len(crystal)):

                ele.append(crystal[i].specie.number)
            atom_fea = np.vstack([self.one_hot_element(ele[i])
                                for i in range(len(crystal))])
          
            struc = torch.Tensor(struc)
            atom_fea = torch.Tensor(atom_fea)
            if not os.path.exists('./h5_data'):
                os.mkdir('h5_data')
            f1 = h5py.File('./h5_data/'+cif_id+'.h5','w')
            f1.create_dataset('atom_fea',data=atom_fea)
            f1.create_dataset('struc',data=struc)
            f1.close()
    def one_hot_element(self,ele):                                                                                                                           
        one_hot=[0, 0, 0, 0, 0, 0, 0, 0, 0,  
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0,  
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0,  
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0,  
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0,  
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0,  
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0,  
             0, 0, 0, 0, 0, 0, 0]  
          
        atomicNum = {'X': 0, 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,  
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
        one_hot[ele]=1
                 
        return one_hot

if __name__=='__main__':
    a = structrue_analy()
    a.analy(opt.xyzpath)
