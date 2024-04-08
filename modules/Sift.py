import torch
from torch import nn
from torch.nn.parameter import Parameter
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from torch.utils.data import Dataset
import pickle, gzip
from torch_sparse import spmm

def sparse2tensor(m):
    """
    Convert sparse matrix (scipy.sparse) to tensor (torch.sparse)
    """
    assert(isinstance(m, sparse.coo.coo_matrix))
    i = torch.LongTensor(np.array([m.row, m.col]))
    v = torch.FloatTensor(m.data)

    return torch.sparse.FloatTensor(i, v, torch.Size(m.shape))

def s2IV(m):
    """
    Convert sparse matrix (scipy.sparse) to tensor (torch.sparse)
    """
    assert(isinstance(m, sparse.coo.coo_matrix))
    i = torch.LongTensor(np.array([m.row, m.col]))
    v = torch.FloatTensor(m.data)

    return i,v

def dense2sparseMM(m,n):
    """
    稠密x稀疏矩阵转稀疏矩阵乘,输出稠密矩阵
    """
    #m稀疏 n稠密
    m = m.squeeze()
    M1 = m.shape[0]
    N1 = m.shape[1]
    M2 = n.shape[0]
    N2 = n.shape[1]
    n=n.T
    m=m.T
    # ED1 = sparse.coo_matrix(m)
    # ED2 = sparse.coo_matrix(n)
    # i2 = torch.nonzero(n).T
    i2 = torch.where(n!=0)
    v2  = n[i2]
    i2 = torch.stack(i2)
    # v2 = n[i2[0],i2[1]]
    out = spmm(i2,v2,N2,M2,m)
    out = out.T
    out = out.unsqueeze(0)
    return out

def Spmm_for_grid2mesh(m,ni,nv,nshape):
    out = []
    for batch in range(m.shape[0]):
        out.append(spmm(ni,nv,nshape[0],nshape[1],m[batch,...]).unsqueeze(0))
    out = torch.cat(out,0)
    return out

def SPmm(m,ni,nv,nshape):
    """
    稠密x稀疏矩阵转稀疏矩阵乘,输出稠密矩阵
    """
    #m稀疏 n稠密
    # bs, feat,10242   x 10242,61440

    bs = m.shape[0]
    m = torch.flatten(m,start_dim=0, end_dim=1)
    # out = []
    # for batch in range(m.shape[0]):
    #     out.append(spmm(ni,nv,nshape[0],nshape[1],m[batch,...].T).T.unsqueeze(0))
    out = spmm(ni,nv,nshape[0],nshape[1],m.T)
    out = out.T
    out = out.view(bs,-1,nshape[0])
    # out = torch.cat(out,0)
    return out

def spmatmul(den, sp):
    """
    den: Dense tensor of shape batch_size x in_chan x #V
    sp : Sparse tensor of shape #V x newlen
    """

    return torch.matmul(den, sp)


def xyz2latlong(vertices):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    long = np.arctan2(y, x)
    xy2 = x**2 + y**2
    lat = np.arctan2(z, np.sqrt(xy2))
    return lat, long

def interp_r2tos2(sig_r2, V, method="linear", dtype=np.float32):
    """
    sig_r2: rectangular shape of (lat, long, n_channels)
    V: array of spherical coordinates of shape (n_vertex, 3)
    method: interpolation method. "linear" or "nearest"
    """
    ele, azi = xyz2latlong(V)
    nlat, nlong = sig_r2.shape[0], sig_r2.shape[1]
    dlat, dlong = np.pi/(nlat-1), 2*np.pi/nlong
    lat = np.linspace(-np.pi/2, np.pi/2, nlat)
    long = np.linspace(-np.pi, np.pi, nlong+1)
    sig_r2 = np.concatenate((sig_r2, sig_r2[:, 0:1]), axis=1)
    intp = RegularGridInterpolator((lat, long), sig_r2, method=method)
    s2 = np.array([ele, azi]).T
    sig_s2 = intp(s2).astype(dtype)
    return sig_s2
    

class Sift(nn.Module):
    def __init__(self, mean_1, mean_2, bias, ncon, mu, recordN, O_1, Dis, k_hop):
        super().__init__()
        self.mean_1 = mean_1
        self.mean_2 = mean_2
        self.bias = bias
        self.ncon = ncon
        self.mu = mu
        self.record = recordN
        self.O_1 = O_1
        self.Dis = Dis
        self.k_hop = k_hop

    def forward(self, x):
        if self.diff_chan or self.coarsen:
            x2 = self.seq2(x) #如果不需要降采样就执行序列2
        else:
            x2 = x
        x1 = self.seq1(x)  #执行序列一
        out = x1 + x2
        out = self.relu(out)
        return out
    
    def HEE(self, x):
        
        self.Dis = 6
        a_1 = self.mean_1
        a_2 = self.mean_2
        ka = 4/(4-torch.pi)
        x = ka*((a_1)**2)*((a_2)**2)*((a_1*a_2-1)**2)*(self.bias)**4
        x = x*self.record/(self.ncon-1)**2
        x /= self.O_1
        x = x*torch.cos(2*torch.pi*self.k_hop/self.Dis)/self.k_hop

        return x