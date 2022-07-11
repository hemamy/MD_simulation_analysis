# reading input files and extract required informtion
import MDAnalysis as mda
import hoomd
import hoomd.md
import numpy as np
import gsd
import gsd.hoomd
import matplotlib
import matplotlib.pyplot as plt
#import imageio
import sys, os
import re
from numpy import linalg as LA
sys.path.insert(1, './')
import functions




def fix_pbc3d_sign(x,l_box):
    l_box_half=l_box/2.0
    #fixing the problem if the x value is slightly larger than the box size
    int_val=np.sign(np.int_(x/l_box_half))
    x=x-l_box*int_val
    return x

def unwrap_pbc_pos(all_pos, univ):
    nlin=all_pos.shape[0]
    nlog=all_pos.shape[1]
    natom=all_pos.shape[2]
    ndim=all_pos.shape[3]
    all_pos=np.reshape(all_pos, (nlin*nlog, natom, ndim))
    fixed=np.empty(0)
    lbox3d=univ.coord.dimensions[0:3]
    for j in range(0,univ.bonds.indices.shape[0]):
        pair=univ.bonds.indices[j]
        if ((np.all(fixed!=pair[0])) and (np.all(fixed!=pair[1]))):
            pair1, pair2 = pair[0], pair[1]
            fixed=np.append(fixed,pair)
            dr=all_pos[:,pair2,:]-all_pos[:,pair1,:]
            dr=functions.fix_pbc3d_sign(dr,lbox3d)
            all_pos[:,pair2,:]=dr+all_pos[:,pair1,:]
        if ((np.any(fixed==pair[0])) and (np.any(fixed==pair[1]))):
            continue

        if ((np.any(fixed==pair[0])) or (np.any(fixed==pair[1]))):

            if(np.any(fixed==pair[0])):
                pair1, pair2 = pair[0], pair[1]
            dr=all_pos[:,pair2,:]-all_pos[:,pair1,:]
            dr=functions.fix_pbc3d_sign(dr,lbox3d)
            all_pos[:,pair2,:]=dr+all_pos[:,pair1,:]

            if(np.any(fixed==pair[1])):
                pair1, pair2 = pair[1], pair[0]
            dr=all_pos[:,pair2,:]-all_pos[:,pair1,:]
            dr=functions.fix_pbc3d_sign(dr,lbox3d)
            all_pos[:,pair2,:]=dr+all_pos[:,pair1,:]

            fixed=np.append(fixed,pair)
            dr=all_pos[:,pair2,:]-all_pos[:,pair1,:]
            dr=functions.fix_pbc3d_sign(dr,lbox3d)
            all_pos[:,pair2,:]=dr+all_pos[:,pair1,:]

    mol_id=np.int_(functions.mol_bond(univ))  
    for i in range(0, np.max(mol_id)+1):
        idx=np.array(np.where(mol_id==i)[0])
        rcom=np.mean(all_pos[:,idx,:], axis=1)
        rcom_new=functions.fix_pbc3d_sign(rcom,lbox3d)
        all_pos[:,idx,:]=np.subtract(all_pos[:,idx,:],np.expand_dims(rcom-rcom_new, axis=1))

    all_pos=np.reshape(all_pos, (nlin, nlog, natom, ndim))
    return(all_pos)

def rouse_mode(all_pos):
    # the posiiton is a 3d array (conf, atom, dim)
    
    pos=np.swapaxes(all_pos,1,2)
    natom=all_pos.shape[1]
    natom_fl=np.float(natom)
    p=np.float_(np.arange(0,natom))
    phi=np.arange(1,natom+1)
    phi=np.float_(phi)
    phi=np.outer(phi-0.5,p)
    phi=phi*np.pi/natom_fl
    pos=np.swapaxes(all_pos,1,2)
    xp=np.matmul(pos, np.cos(phi))/natom_fl
    xp=np.swapaxes(xp,1,2)
    return(xp)

def corr_lin(data, log_space):
    nlin=data.shape[0]
    nlog=data.shape[1]
    x=data.copy()
    norm=np.mean(np.sum(x[:,:,:]*x[:,:,:], axis=2),axis=0)
    corr=[]
    error=[]
    for i in range (1,nlin):
        norm=np.concatenate((x[i:,:,:]*x[i:,:,:],x[:-i,:,:]*x[:-i,:,:]), axis=0)
        norm=np.mean(norm,axis=(0,1))
        norm=np.sum(norm,axis=0)
        add=np.sum(x[i:,:,:]*x[:-i,:,:],axis=2)/norm
        error_add=np.std(add)/np.sqrt(x[i:,:,:].shape[0]*x[i:,:,:].shape[1])
        add=np.mean(add,axis=(0,1))
        corr.append(add)
        error.append(error_add)
        
    corr=np.array(corr)
    error=np.array(error)
    time=log_space**(nlog)*np.arange(1, nlin)+1# +1 because each block has (2**N)+1 timesteps
    time=np.array(time)
    if(log_space==1):
        time=np.arange(1, nlin)
    return(corr, time, error)

def corr_log(data, log_space):
    #data is nlin x nlog x 1 array (nlin, nlog, value)
    nlin=data.shape[0]
    nlog=data.shape[1]
    x=data.copy()
    norm=x[:,:,:]*x[:,:,:]
    norm=np.mean(norm, axis=0)
    norm=np.sum(norm, axis=1)
    corr=np.sum(x[:,:,:]*x[:,:1,:],axis=2)
    corr=np.where(norm == 0, norm, corr/norm)
    error=np.std(corr, axis=0)/np.sqrt(len(corr))
    corr=np.mean(corr, axis=0)
    time=log_space**(np.arange(1, nlog+1))-log_space
    time=np.array(time)
    return(corr, time, error)

def fixing_pbc(x,l_box):
    l_box_half=l_box/2.0
    x=x-l_box*np.int_(x/l_box_half)
    return x

def read_gsd(filename):
    return(gsd.hoomd.open(filename, 'rb'))

def write_gsd(snapshot, filename):
    f = gsd.hoomd.open(name=filename, mode='wb')
    f.append(snapshot)
    f.close()
    
def read_univ(filename):
    univ= mda.Universe(filename)
    return(mda.Universe(filename))

def read_univ_topo(dcdfile, topofile):
    return(mda.Universe(topofile, dcdfile))

def univ_pos(univ):
    pos=[]
    for i, v in enumerate(univ.trajectory):
        temp=v.positions.copy()
        pos.append(temp)
    pos=np.array(pos)
    return(pos)

def read_univ_topo(FileName, TopoFileName):
    univ= mda.Universe(FileName, TopoFileName)
    return(mda.Universe(FileName, TopoFileName))

#funcitons

def removing_com_drift(x,l_box):
    dR=np.zeros(x.shape[0])
    drift=np.zeros((x.shape[0],x.shape[2]))
    for t in range(1, x.shape[0]):
        dr=x[t,:,:]-x[t-1,:,:]
        #functions.fix_pbc3d(dr,l_box)
        functions.fix_pbc3d_sign(dr,l_box)
        drift[t,:]=np.sum(dr,axis=0)
        drift[t,:]=drift[t,:]/x.shape[1]
        x[t,:,:]=x[t,:,:]-drift[t,:]
        
    for t in range(1, 2):
        functions.fix_pbc3d(x[t,:,:],l_box)





# msd 

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
                    alpha2[tw]+=np.sum(r2*r2)/(float(x.shape[1])*float(dr_tw.shape[0]))
        sys.stdout.write("\rcompleated=%d"% (int(100.*log_id/nlog)))
        sys.stdout.flush()



                
    #taking average
    msd=msd/(nlog)
    alpha2=alpha2/(nlog)
    time=(log_space**(nlog)+1.)*np.arange(1, nlin)
    if(log_space==1):
        time=np.arange(1, nlin)
 
    return(msd,(3.*alpha2)/(5.*msd*msd)-1, time )




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
            if(np.any(dr>lbox[:])):
                print("after",np.max(dr),lbox[:])
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

# lin-spaced data


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
        error.append(np.std(fqt_list[tw])/np.sqrt(len(fqt_list[tw])))
    time=(log_space**(nlog)+1.)*np.arange(1, nlin)
    if(log_space==1):
        time=np.arange(1, nlin)
    return(fqt_out, time, error, cal)



def mol_bond(univ_topo):
    natoms=univ_topo.atoms.n_atoms
    bond_id=np.zeros(natoms)-1
    bond_info=np.array(univ_topo.bonds.indices)
    mol_id=0
    for i in range(0,univ_topo.bonds.indices.shape[0]):
        id0=bond_info[i,0]
        id1=bond_info[i,1]

        #check if the atoms are already belong to a molecules

        if(bond_id[id0]!=-1):
            bond_id[id1]=bond_id[id0]
        if(bond_id[id1]!=-1):
            bond_id[id0]=bond_id[id1]
        if(bond_id[id0]==-1 and bond_id[id1]==-1):
            bond_id[id0],bond_id[id1]=mol_id,mol_id
            mol_id=mol_id+1

    return(bond_id) 

def gTensor(pos, lbox, mol_id):
    NDIM=3
    gtensor=np.zeros((1,3,3))
    gtensor_temp=np.zeros((3,3))
    for t in range(pos.shape[0]):
        for m in range(0, np.max(mol_id)+1):

            deltar=[]
            idx = np.where(mol_id==m)
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


def bond_orientation_relaxation_lin(all_pos, lbox, mol_id, log_space):
    nlin=all_pos.shape[0]
    nlog=all_pos.shape[1]
    natom=all_pos.shape[2]
    corr=np.zeros(nlin-1)
    bo_out=[]
    error=[]
    corr=[]
    corr= [[[None] for _ in range(0)] for _ in range(nlin-1)]
    count=0
    cal=0
    for log_id in range(0, nlog):
        x=all_pos[:,log_id,:,:]
        for m in range(0, np.max(mol_id)+1):
            idx = np.where(mol_id==m)[0]
            mol_pos = x[:,idx,:]
            for i in range(0, mol_pos.shape[1]-1):
                dr = mol_pos[:,i,:]-mol_pos[:,i+1,:]
                dr=functions.fix_pbc3d_sign(dr, lbox)
                dr_sum = np.cumsum(dr, axis=0)
                cal=1
                for tw in range(1,nlin):
                    if(tw==nlin-1):
                        corr[tw-1].append(np.sum(dr[tw:,:]*dr[:-tw,:], axis=1)/np.mean(np.sum(dr*dr, axis=1)))
                    else:
                        corr[tw-1].append(np.sum(dr[tw:,:]*dr[:-tw,:], axis=1)/np.mean(np.sum(dr*dr, axis=1)))
                count+=1

    for tw in range(0,nlin-1):
        bo_out.append(np.mean(corr[tw]))
        error.append(np.std(corr[tw])/np.sqrt(sum(len(x) for x in corr[tw])))
    bo_out, error= np.array(bo_out), np.array(error)
    time=(log_space**(nlog)+1.)*np.arange(1, nlin)
    if(log_space==1):
        time=np.arange(1, nlin)
    return(bo_out, time, error, cal)


def sq(QMAX, binsize, L, pos_traj, ti, tf):
    qmin=2.0*np.pi/L
    sq=np.zeros(QMAX)
    sq_avg=np.zeros(QMAX)
    count_tot=np.zeros(QMAX)
    q_vec=np.zeros(QMAX)
    count=0
    for t in range(ti,tf):
        pos=pos_traj[t,:,:]
        sq=np.zeros(QMAX)
        for q in range(2, QMAX):
            filename="qvectors/qvector.%03d" % q
            qvec=np.loadtxt(filename)
            q_vec[q]=q*qmin/2.0
            qvec=qmin*qvec
            #reduce the number of calculated qs in each bin
            if (qvec.shape[0]>50):
                qvec= qvec[0:50,:]

            for j in range(0, qvec.shape[0]):
                rhoq_r, rhoq_i, = 0, 0
                for i in range (0,pos.shape[0]):
                    qdotr=np.dot(qvec[j,:], pos[i,:])
                    qdotr=qvec[j][0]*pos[i][0] + qvec[j][1]*pos[i][1] + qvec[j][2]*pos[i][2]
                    rhoq_r = rhoq_r + np.cos(qdotr)
                    rhoq_i = rhoq_i + np.sin(qdotr)
                sq[q] = sq[q] + (rhoq_r*rhoq_r + rhoq_i*rhoq_i)/pos.shape[0]
            sq[q]=sq[q]/qvec.shape[0]
            sq_avg[q]=sq_avg[q]+sq[q]


    sq_avg[:]=sq_avg[:]/pos_traj.shape[0]
    return(sq_avg, q_vec)









