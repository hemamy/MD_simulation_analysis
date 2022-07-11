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
    data = []
    nlin = len(filename)
    for file_n in filename:
        with open(file_n) as fi:
            data.append(fi.read())
    nlog = len(data[0].split('ITEM: TIMESTEP\n'))-1
    data = "".join(data)
    #extract initial information and remove the box information text from the coordinates data
    data = data.split('ITEM: TIMESTEP\n')
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
        data[i] = 0
    #join all of the timesteps in a single list
    data_temp = "".join(data_temp)

    nfield = len(data_temp.split('\n')[0].split())
    natom = len(data_temp.split('\n\n')[0].split())
    natom = int(natom/nfield)

    #creat a numpy array from the imported data
    data=data_temp.split('\n\n')
    data_temp = []
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




def msd_lin(all_pos,lbox,log_space):
    nlin=all_pos.shape[0]
    nlog=all_pos.shape[1]
    natom=all_pos.shape[2]
    msd=np.zeros(nlin-1)
    alpha2=np.zeros(nlin-1)
    x=all_pos.copy()
    x=np.reshape(x,(int(x.shape[0])*int(x.shape[1]), int(x.shape[2]), int(x.shape[3])))
    xnew=np.zeros_like(x)
    for i in range(0, natom):
        dr = x[1:,i,:]-x[:-1,i,:]
        #if(np.any(dr>lbox[2])):
            #print(np.max(dr),lbox[:], )
        dr=functions.fix_pbc3d_sign(dr, lbox)
        if(np.any(dr>lbox[:])):
            print("after",np.max(dr),lbox[:])
        dr_sum_tot = np.cumsum(dr, axis=0)
        x[1:,i,:]=dr_sum_tot + x[0,i,:]
    x=np.reshape(x,(int(all_pos.shape[0]),int(all_pos.shape[1]), int(all_pos.shape[2]), int(all_pos.shape[3])))
    x_orig=x.copy()

    for log_id in range(0, nlog):
        x=x_orig[:,log_id,:,:]
        for i in range(0, x.shape[1]):
            dr = x[1:,i,:]-x[:-1,i,:]
            dr_sum = np.cumsum(dr, axis=0)

            #print(dr)
            for tw in range(0,nlin-1):
                if(tw==nlin-2):
                    dr_tw=dr_sum[tw-1:tw,:]
                    r2=np.sum(dr_tw*dr_tw, axis=1)
                    msd[tw]+=np.sum(r2)/(float(x.shape[1])*float(dr_tw.shape[0]))
                    alpha2[tw]+=np.sum(r2*r2)/(float(x.shape[1])*float(dr_tw.shape[0]))
                else:
                    dr_tw=dr_sum[tw+1:,:] - dr_sum[:-tw-1,:]
                    r2=np.sum(dr_tw*dr_tw, axis=1)
                    msd[tw]+=np.sum(r2)/(float(x.shape[1])*float(dr_tw.shape[0]))
                    #print(tw,msd[tw-1])
                    alpha2[tw]+=np.sum(r2*r2)/(float(x.shape[1])*float(dr_tw.shape[0]))
        sys.stdout.write("\rcompleated=%d"% (int(100.*log_id/nlog)))
        sys.stdout.flush()




    #taking average
    msd=msd/(nlog)
    alpha2=alpha2/(nlog)
    time=(log_space**(nlog)+1.)*np.arange(1, nlin)
    if(log_space==1):
        time=np.arange(1, nlin)

    return(msd,(3.*alpha2)/(5.*msd*msd)-1, time)



def msd_log(all_pos,lbox,log_space):
    nlin=all_pos.shape[0]
    nlog=all_pos.shape[1]
    natom=all_pos.shape[2]
    msd=np.zeros(nlog-1)
    alpha2=np.zeros(nlog-1)
    for lin_id in range(0, nlin):
        x=all_pos[lin_id,:,:,:]
        for i in range(0, x.shape[1]):
            dr = x[1:,i,:]-x[:-1,i,:]
            dr=functions.fix_pbc3d_sign(dr[:], lbox)
            #dr=functions.fix_pbc3d(dr, lbox)
            if(np.any(dr>lbox[:])):
                print("after",np.max(dr),lbox[:])
            #dr=np.cumsum(dr)
            dr_sum = np.cumsum(dr, axis=0)
            for tw in range(0,1):
                if(tw==0):
                    dr_tw=dr_sum.copy()
                    r2=np.sum(dr_tw*dr_tw, axis=1)
                    msd+=r2/(float(x.shape[1]))
                    alpha2+=r2*r2/(float(x.shape[1]))
        sys.stdout.write("\rcompleated=%d"% (int(100.*lin_id/nlin)))
        sys.stdout.flush()
        #taking average
    msd=msd/(nlin)
    alpha2=alpha2/(nlin)
    time=log_space**(np.arange(2, nlog+1))-log_space
    return(msd,(3.*alpha2)/(5.*msd*msd)-1, time )


















