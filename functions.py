# reading input files and extract required informtion
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import imageio
import sys, os
import re
from numpy import linalg as LA
sys.path.insert(1, './')
import functions


def read_lammps(filename):
    content = []
    nlin = len(filename)
    for file_n in filename:
        with open(file_n) as fi:
            content.append(fi.read())
    nlog = len(content[0].split('ITEM: TIMESTEP\n'))-1
    content = "".join(content)
    #extract initial information and remove the box information text from the coordinates data
    data = content.split('ITEM: TIMESTEP\n')
    data = data[1:]
    timestep = []
    data_temp = []
    boxsize = np.zeros((len(data),3))
    for i in range(0,len(data)):
        timestep.append(data[i].split('\n')[0])
        boxsize[i][0] = float(data[i].split('\n')[4].split()[1])-float(data[i].split('\n')[4].split()[0])
        boxsize[i][1] = float(data[i].split('\n')[5].split()[1])-float(data[i].split('\n')[5].split()[0])
        boxsize[i][2] = float(data[i].split('\n')[6].split()[1])-float(data[i].split('\n')[6].split()[0])
        #depending on the dump file structure, "vz\ " should be replaced by proper string
        data_temp += (data[i].split('vz \n')[1])+"\n"
    #join all of the timesteps in a single list
    data_temp = "".join(data_temp)

    nfield = len(data_temp.split('\n')[0].split())
    natom = len(data_temp.split('\n\n')[0].split())
    natom = int(natom/nfield)

    #creat a numpy array from the imported data
    data=data_temp.split('\n\n')
    # removes the last empty element
    data=data[0:-1]
    temp = [conf.split('\n')  for conf in data]
    pos = []
    for conf in data:
        temp = conf.split('\n')
        pos_conf = []
        for line in temp:
            pos_conf.append(line.split(' '))
        pos.append(pos_conf)
    pos = np.array(pos)
    pos = pos
    del temp, data, data_temp

    return (pos, nlin, nlog, boxsize, timestep)







def fix_pbc3d_sign(x,l_box):
    l_box_half=l_box/2.0
    #fixing the problem if the x value is slightly larger than the box size
    int_val=np.sign(np.int_(x/l_box_half))
    x=x-l_box*int_val
    return x





def removing_com_drift(x,l_box):
    dR=np.zeros(x.shape[0])
    drift=np.zeros((x.shape[0],x.shape[2]))
    for t in range(1, x.shape[0]):
        dr = x[t,:,:] - x[t-1,:,:]
        #functions.fix_pbc3d(dr,l_box)
        dr = functions.fix_pbc3d_sign(dr,l_box)
        drift[t,:]=np.mean(dr,axis=0)
    tot_drift = np.cumsum(drift, axis = 0)
    tot_drift = tot_drift - np.int_(tot_drift/l_box)
    for t in range(1, x.shape[0]):
        x[t,:,:] = x[t,:,:] - tot_drift[t,:]
        x[t,:,:] = functions.fix_pbc3d_sign(x[t,:,:],l_box)
    return(x)



def gTensor(pos, lbox, mol_id):
    NDIM=3
    gtensor=np.zeros((1,3,3))
    gtensor_temp=np.zeros((3,3))
    for t in range(pos.shape[0]):
        for m in range(0, np.max(mol_id[t,:])+1):

            deltar=[]
            idx = np.where(mol_id[t,:]==m)
            mol_pos = pos[t][idx][:]
            for k in range(NDIM):
                deltar.append(mol_pos[np.newaxis,:,k]-mol_pos[:,k,np.newaxis])
            deltar=np.array(deltar)
            deltar=np.swapaxes(deltar, 0,2)
            deltar=np.swapaxes(deltar, 0,1)

            #fixing PBC
            temp=deltar.copy()
            temp=np.reshape(temp, (deltar.shape[0] * deltar.shape[1], deltar.shape[2]))
            temp=functions.fix_pbc3d_sign(temp,lbox)
            #temp=functions.fix_pbc3d(temp,lbox)
            temp=np.reshape(temp, (deltar.shape[0], deltar.shape[1], deltar.shape[2]))
            deltar=temp.copy()
            for i in range(0,NDIM):
                for j in range(0,NDIM):
                    gtensor_temp[i][j]=np.sum(deltar[:,:,i]*deltar[:,:,j])
            gtensor=np.append(gtensor, gtensor_temp[np.newaxis,:,:], axis=0)
    gtensor=np.delete(gtensor,0, axis=0)/(2.* mol_pos.shape[0]**2.)
    return(gtensor)


def rg(gtensor):
    NDIM=3
    rg_tot=[]
    eigenvec_all = np.zeros((NDIM,NDIM,1))
    eigenval_all = np.zeros((NDIM,1))
    # number of total polymer chains calculated in gyration.dat file
    N = int(gtensor.shape[0]/NDIM)
    # reshaping the arrays
    gtensor = np.reshape(gtensor, (N, NDIM, NDIM))
    #begining calculations
    for i in range(0, N):
        eigenval, eigenvec = LA.eig(gtensor[i, :, :])
        idx = eigenval.argsort()[::-1]
        eigenval = eigenval[idx]
        eigenvec = eigenvec[:, idx]

        eigenvec_reshape = np.reshape(eigenvec , (NDIM,NDIM,1))
        eigenvec_all = np.append(eigenvec_all, eigenvec_reshape, axis = 2)

        eigenval_reshape = np.reshape(eigenval , (NDIM,1))
        eigenval_all = np.append(eigenval_all, eigenval_reshape, axis = 1)

        rg = eigenval[0] + eigenval[1] + eigenval[2]
        rg_tot.append(rg)
    eigenval_all=np.delete(eigenval_all,0, axis=1)
    eigenvec_all = np.delete(eigenvec_all, 0, axis = 2)
    return(np.sqrt(np.array(rg_tot)),eigenval_all, eigenvec_all)

















