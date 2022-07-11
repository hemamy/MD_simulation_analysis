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
        sys.stdout.write("\rcompleated=%d"% (int(100.*i/len(data))))
        sys.stdout.flush()
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
    int_val=np.round_(x/l_box) # round the x/l_box 0 for the distance less than 0.5 and 1 for larger 
    
    if (np.max(int_val)>1. or np.min(int_val)<-1):
        print(np.max(int_val), np.min(int_val))
    x=x-l_box*int_val
    return x


def unwrap_pos(x,l_box):
    # reshaping the array of positions to [conf][atomid][pos]
    reshape_flag = 0
    if(len(x.shape)!=3 and  len(x.shape)!=4):
        print("array shape should be (t_lin, t_log, id, pos) or (t, id, pos)")
    if(len(x.shape)==4):
        nlin = x.shape[0]; nlog = x.shape[1]; natoms = x.shape[2]; ndims = x.shape[3];
        x = np.reshape(x, (nlin * nlog, natoms, ndims))
        reshape_flag =1
    # calculate  the dr between every two consecutive snapshots
    dr = x[1:,:,:] - x[:-1,:,:]
    print(np.max(dr), np.min(dr))
    # fixing the periodic boundary condition 
    dr = functions.fix_pbc3d_sign(dr,l_box)
    dr = np.cumsum(dr, axis=0) # create a cumulative sum of the dr values
    #dr = functions.fix_pbc3d(dr,l_box)
    #updating the positions of the particles
    x = np.append(x[0:1,:,:], x[0,:,:] + dr, axis=0)
    # reshaping back the array again
    if (reshape_flag==1):
        x = np.reshape(x, (nlin, nlog, natoms, ndims))
    return(x)
        
    






def removing_com_drift(x,l_box):
    dR=np.zeros(x.shape[0])
    drift=np.zeros((x.shape[0],x.shape[2]))
    for t in range(1, x.shape[0]): # looping over all configurations
        dr = x[t,:,:] - x[t-1,:,:] # displacement of the particles between two  snapshot
        #dr = functions.fix_pbc3d_sign(dr,l_box) # uncomment if the positions are not unwraped
        drift[t,:]=np.mean(dr,axis=0)
    tot_drift = np.cumsum(drift, axis = 0) # summing over all of the drifts to calculate the total drift
    #tot_drift = tot_drift - np.int_(tot_drift/l_box) # uncomment if the positions are not unwraped
    for t in range(1, x.shape[0]):
        x[t,:,:] = x[t,:,:] - tot_drift[t,:] 
        #x[t,:,:] = functions.fix_pbc3d_sign(x[t,:,:],l_box)
    return(x)



def gTensor(pos, lbox, mol_id): # mole_id is the id of the molecule that each particle blongs to. array size[natoms]
    reshape_flag = 0
    if(len(pos.shape)!=3 and  len(pos.shape)!=4):
        print("array shape should be (t_lin, t_log, id, pos) or (t, id, pos)")
    if(len(pos.shape)==4):
        nlin = pos.shape[0]; nlog = pos.shape[1]; natoms = pos.shape[2]; ndims = pos.shape[3];
        pos = np.reshape(pos, (nlin * nlog, natoms, ndims))
        reshape_flag =1
    NDIM=3
    gtensor=np.zeros((1,3,3))
    gtensor_temp=np.zeros((3,3))
    for t in range(pos.shape[0]): # vectorization f the deltar calculation for all snapshots  and particles
        for m in range(0, np.max(mol_id[t,:])+1):

            deltar=[]
            idx = np.where(mol_id[t,:]==m)
            mol_pos = pos[t][idx][:]
            for k in range(NDIM):
                deltar.append(mol_pos[np.newaxis,:,k]-mol_pos[:,k,np.newaxis]) # calulate the delta r array between all of the particles in the molecule using the array brodcasting method 
            deltar=np.array(deltar)
            deltar=np.swapaxes(deltar, 0,2)
            deltar=np.swapaxes(deltar, 0,1) # rearranging the order of the delta r array (x,y and z) dims are in the last aray's dim

#            #fixing PBC
#            temp=deltar.copy()
#            temp=np.reshape(temp, (deltar.shape[0] * deltar.shape[1], deltar.shape[2]))
#            temp=functions.fix_pbc3d_sign(temp,lbox)
#            #temp=functions.fix_pbc3d(temp,lbox)
#            temp=np.reshape(temp, (deltar.shape[0], deltar.shape[1], deltar.shape[2]))
#            deltar=temp.copy()
            for i in range(0,NDIM):
                for j in range(0,NDIM):
                    gtensor_temp[i][j]=np.sum(deltar[:,:,i]*deltar[:,:,j]) # calculating the gyration matrix for all snapshots and particles
            gtensor=np.append(gtensor, gtensor_temp[np.newaxis,:,:], axis=0)
    gtensor=np.delete(gtensor,0, axis=0)/(2.* mol_pos.shape[0]**2.)
    if (reshape_flag==1):
        pos = np.reshape(pos, (nlin, nlog, natoms, ndims))
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




def msd_lin(pos,lbox,log_space):
    NDIM = 3
    if(len(pos.shape)!=4):
        print("array shape should be (t_lin, t_log, id, pos)")

    nlin = pos.shape[0]; nlog = pos.shape[1]; natoms = pos.shape[2]; ndims = pos.shape[3];
    msd=np.zeros(nlin-1)
    alpha2=np.zeros(nlin-1)
    x=np.reshape(pos, (nlin*nlog, natoms, NDIM))
    for i in range(0, natoms):
        dr = x[1:,i,:]-x[:-1,i,:]
        dr_sum_tot = np.cumsum(dr, axis=0)
        x[1:,i,:]=dr_sum_tot + x[0,i,:]
    x=np.reshape(x,(nlin, nlog, natoms, NDIM))
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



def msd_log(pos,lbox,log_space):
    NDIM = 3
    if(len(pos.shape)!=4):
        print("array shape should be (t_lin, t_log, id, pos)")

    nlin = pos.shape[0]; nlog = pos.shape[1]; natoms = pos.shape[2]; ndims = pos.shape[3];
    msd=np.zeros(nlog-1)
    alpha2=np.zeros(nlog-1)
    for lin_id in range(0, nlin):
        x=pos[lin_id,:,:,:]
        for i in range(0, x.shape[1]):
            dr = x[1:,i,:]-x[:-1,i,:]
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



def fqt_self_lin(all_pos,lbox,Q, log_space):

    nlin=all_pos.shape[0]
    nlog=all_pos.shape[1]
    natom=all_pos.shape[2]
    fqt=np.zeros(nlin-1)
    fqt_list= [[[] for _ in range(0)] for _ in range(nlin-1)]
    fqt_out=[]
    error=[]
    qmin = 2.0*np.pi/lbox[0]
    q = int(2.*Q/qmin-1.)
    filename="qvectors/qvector.%03d" % q
    qvec=np.loadtxt(filename)
    qvec=qvec*qmin
    qvect=np.transpose(qvec)
    cal=0
    for log_id in range(0, nlog):
        x=all_pos[:,log_id,:,:]

        for i in range(0, x.shape[1]):
            dr = x[1:,i,:]-x[:-1,i,:]
            functions.fix_pbc3d_sign(dr[:], lbox)
            #functions.fix_pbc3d(dr[:], lbox)
            #dr=np.cumsum(dr)
            dr_sum = np.cumsum(dr, axis=0)
            for tw in range(0,nlin-1):
                if(tw==0):
                    dr_tw=dr
                    temp=np.matmul(dr_tw,qvect)
                    temp=np.cos(temp)
                    temp=np.sum(temp, axis=1)/float(qvect.shape[1])
                    #temp=list(temp)
                    fqt_list[tw].append(temp)

                else:
                    dr_tw=dr_sum[tw:,:] - dr_sum[:-tw,:]
                    temp=np.matmul(dr_tw,qvect)
                    temp=np.cos(temp)
                    temp=np.sum(temp, axis=1)/float(qvect.shape[1])
                    fqt_list[tw].append(temp)
                    cal=1
        sys.stdout.write("\rcompleated=%d"% (int(100.*log_id/nlog)))
        sys.stdout.flush()
    for tw in range(0,nlin-1):
        #fqt_out.append(np.mean(np.array(fqt_list[tw]), axis=(0,1)))
        fqt_out.append(np.mean(np.array(fqt_list[tw])))
        #error.append(np.std(fqt_list[tw], axis=(0,1))/np.sqrt(len(fqt_list[tw])))
        #error.append(np.std(fqt_list[tw])/np.sqrt(len(fqt_list[tw])))
    time=(log_space**(nlog)+1.)*np.arange(1, nlin)
    if(log_space==1):
        time=np.arange(1, nlin)
    #return(fqt_out, time, error, cal)
    return(np.array(fqt_out), time)

def fqt_self_log(all_pos,lbox,Q, log_space):
    nlin=all_pos.shape[0]
    nlog=all_pos.shape[1]
    natom=all_pos.shape[2]
    fqt=np.zeros(nlog-1)
    qmin = 2.0*np.pi/lbox[0]
    q = int(2.*Q/qmin-1.)
    filename="./qvectors/qvector.%03d" % q
    qvec=np.loadtxt(filename)
    qvec=qvec*qmin
    qvect=np.transpose(qvec)
    fqt_list=[]
    fqt_out=[]
    error=[]
    for lin_id in range(0, nlin):
        x=all_pos[lin_id,:,:,:]
        for i in range(0, 5):
            dr = x[1:,i,:]-x[:-1,i,:]

            for tw in range(0,1):
                if(tw==0):
                    temp=np.matmul(dr,qvect)
                    temp=np.cos(temp)
                    temp=np.mean(temp, axis=1) # averaging over all qvectors
                    fqt_list.append(list(temp)) # appending the values for each particle
                    cal=1
        sys.stdout.write("\rcompleated=%d"% (int(100.*lin_id/nlin)))
        sys.stdout.flush()
    fqt_list = np.array(fqt_list)
    time=log_space**(np.arange(2, nlog+1))-log_space
    return(np.mean(fqt_list, axis=0), time)




# log-spaced data
def fqt_self_log_2(all_pos,lbox,Q, log_space):
    nlin=all_pos.shape[0]
    nlog=all_pos.shape[1]
    natom=all_pos.shape[2]
    fqt=np.zeros(nlog-1)
    qmin = 2.0*np.pi/lbox[0]
    q = int(2.*Q/qmin-1.)
    filename="./qvectors/qvector.%03d" % q
    qvec=np.loadtxt(filename)
    qvec=qvec*qmin
    qvect=np.transpose(qvec)
    for lin_id in range(0, nlin):
        x=all_pos[lin_id,:,:,:]

        for i in range(0, x.shape[1]):
            dr = x[1:,i,:]-x[:-1,i,:]
            for tw in range(0,1):
                if(tw==0):
                    temp=np.matmul(dr,qvect)
                    temp=np.cos(temp)
                    temp=np.sum(temp, axis=1)
                    fqt+=temp
        sys.stdout.write("\rcompleated=%d"% (int(100.*lin_id/nlin)))
        sys.stdout.flush()
    time=log_space**(np.arange(2, nlog+1))-log_space
    return(fqt/(qvec.shape[0]*x.shape[1]*nlin), time)








