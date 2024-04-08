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

import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from itertools import chain
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parallel.data_parallel import _check_balance
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.cuda._utils import _get_device_index

class NodeUpdate(nn.Module):
  def __init__(self, layer_id, in_feats, out_feats, dropout, activation=None, test=False, concat=False):
    super(NodeUpdate, self).__init__()
    self.layer_id = layer_id
    self.linear = nn.Linear(in_feats, out_feats)
    self.dropout = None
    if dropout != 0:
        self.dropout = nn.Dropout(p=dropout)
    self.activation = activation
    self.concat = concat
    self.test = test

  def forward(self, node):
    h = node.data['h']
    if self.test:
        norm = node.data['norm']
        h = h * norm
    else:
        agg_history_str = 'agg_h_{}'.format(self.layer_id-1)
        agg_history = node.data[agg_history_str]
        # control variate
        h = h + agg_history
        if self.dropout:
            h = self.dropout(h)
    h = self.linear(h)
    if self.concat:
        h = torch.cat((h, self.activation(h)), dim=1)
    elif self.activation:
        h = self.activation(h)
    return {'activation': h}


class GCNSampling(nn.Module):
  def __init__(self,
               in_feats,
               n_hidden,
               n_classes,
               n_layers,
               activation,
               dropout):
    super(GCNSampling, self).__init__()
    self.n_layers = n_layers
    self.dropout = None
    self.preprocess = False
    if dropout != 0:
        self.dropout = nn.Dropout(p=dropout)
    self.activation = activation
    # input layer
    self.linear = nn.Linear(in_feats, n_hidden)
    self.layers = nn.ModuleList()
    # hidden layers
    for i in range(1, n_layers):
        skip_start = (i == n_layers-1)
        self.layers.append(NodeUpdate(i, n_hidden, n_hidden, dropout, activation, concat=skip_start))
    # output layer
    self.layers.append(NodeUpdate(n_layers, 2*n_hidden, n_classes, dropout))

  def forward(self, nf):

    h = nf.layers[0].data['preprocess']
    if self.dropout:
        h = self.dropout(h)
    h = self.linear(h)

    skip_start = (0 == self.n_layers-1)
    if skip_start:
      h = torch.cat((h, self.activation(h)), dim=1)
    else:
      h = self.activation(h)

    for i, layer in enumerate(self.layers):
      new_history = h.clone().detach()
      history_str = 'h_{}'.format(i)
      history = nf.layers[i].data[history_str]
      h = h - history

      nf.layers[i].data['h'] = h
      nf.block_compute(i,
                       fn.copy_src(src='h', out='m'),
                       fn.mean(msg='m', out='h'),
                       layer)
      h = nf.layers[i+1].data.pop('activation')
      # update history
      if i < nf.num_layers-1:
          nf.layers[i].data[history_str] = new_history

    return h

from dgl import NodeFlow
from dgl.contrib.sampling import NeighborSampler

class DGLNodeFlowLoader():
  """
  Generate inputs data and labels at each iteration.
  inputs: will be a list of dgl.NodeFlows
          whose length is equal to `torch.cuda.device_count()`.
  labels: will be a tensor which concats all labels 
          corresponded to nodeflows in the inputs
  Note:
    Current implementation only supports 
      `dgl.contrib.sampling.NeighborSampler`
  """
  def __init__(self, graph, labels, batch_size,
               num_hops, seed_nodes, sample_type='neighbor',
               num_neighbors=8, num_worker=32):
    self.graph = graph
    self.labels = labels
    self.batch_size = batch_size
    self.type = sample_type
    self.num_hops = num_hops
    self.seed_nodes = seed_nodes
    self.num_neighbors = num_neighbors
    self.num_worker = num_worker

    self.device_num = torch.cuda.device_count()
    if self.device_num == 0:
      self.device_num = 1 # cpu
    per_worker_batch = int(self.batch_size / self.device_num)
    if self.type == "neighbor":
      self.sampler = NeighborSampler(self.graph,
                                     per_worker_batch,
                                     self.num_neighbors,
                                     neighbor_type='in',
                                     shuffle=True,
                                     num_workers=self.num_worker,
                                     num_hops=self.num_hops,
                                     seed_nodes=self.seed_nodes,
                                     prefetch=True)
    else:
      self.sampler = None
      raise RuntimeError("Currently only support Neighbor Sampling")
    
    self.sampler_iter = None
  
  def __iter__(self):
    self.sampler_iter = iter(self.sampler)
    return self

  def __next__(self):
    nf_list = []
    label_list = []
    for i in range(self.device_num):
      try:
        nf = next(self.sampler_iter)
        batch_nids = nf.layer_parent_nid(-1)
        nf_list.append(nf)
        label_list.append(self.labels[batch_nids])
      except StopIteration:
        if len(nf_list) == 0:
          raise StopIteration
        else: # the last batch
          break
    labels = torch.cat(label_list)
    return nf_list, labels

class AME(nn.Module):
    '''Asynchronous Message Engine'''
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation):
        super(AME, self).__init__()
        self.n_layers = n_layers
        self.activation = activation
        # input layer
        self.linear = nn.Linear(in_feats, n_hidden)
        self.layers = nn.ModuleList()
        # hidden layers
        for i in range(1, n_layers):
            skip_start = (i == n_layers-1)
            self.layers.append(NodeUpdate(i, n_hidden, n_hidden, 0, activation, True, concat=skip_start))
        # output layer
        self.layers.append(NodeUpdate(n_layers, 2*n_hidden, n_classes, 0, None, True))

    def forward(self, x):
        h = x.layers[0].data['preprocess']
        h = self.linear(h)

        skip_start = (0 == self.n_layers-1)
        if skip_start:
            h = torch.cat((h, self.activation(h)), dim=1)
        else:
            h = self.activation(h)

        for i, layer in enumerate(self.layers):
            x.layers[i].data['h'] = h
            x = self.GETK(h)
            x = self.SR(x)
            x.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'),
                             layer)
            h = x.layers[i+1].data.pop('activation')

        return h
    
    def GETK(self, x):

        h = x.layers[0].data['preprocess']
        if self.dropout:
            h = self.dropout(h)
        h = self.linear(h)

        skip_start = (0 == self.n_layers-1)
        if skip_start:
            h = torch.cat((h, self.activation(h)), dim=1)
        else:
            h = self.activation(h)

        for i, layer in enumerate(self.layers):
            new_history = h.clone().detach()
            history_str = 'h_{}'.format(i)
            history = nf.layers[i].data[history_str]
            h = h - history

        x.layers[i].data['h'] = h
        x.block_compute(i,
                        fn.copy_src(src='h', out='m'),
                        fn.mean(msg='m', out='h'),
                        layer)
        h = x.layers[i+1].data.pop('activation')
        # update history
        if i < x.num_layers-1:
            x.layers[i].data[history_str] = new_history

        return h
    
    def SR(self, nf):
        if self.preprocess:
        for i in range(nf.num_layers):
            h = nf.layers[i].data.pop('features')
            neigh = nf.layers[i].data.pop('neigh')
            if self.dropout:
                h = self.dropout(h)
                h = self.fc_self(h) + self.fc_neigh(neigh)
                skip_start = (0 == self.n_layers - 1)
            if skip_start:
                h = torch.cat((h, self.activation(h)), dim=1)
            else:
                h = self.activation(h)
                nf.layers[i].data['h'] = h
        else:
            for lid in range(nf.num_layers):
                nf.layers[lid].data['h'] = nf.layers[lid].data.pop('features')

        for lid, layer in enumerate(self.layers):
            for i in range(lid, nf.num_layers - 1):
                h = nf.layers[i].data.pop('h')
                h = self.dropout(h)
                nf.layers[i].data['h'] = h
                if self.aggregator_type == 'mean':
                    nf.block_compute(i,
                                    fn.copy_src(src='h', out='m'),
                                    fn.mean('m', 'neigh'),
                                    layer)
                elif self.aggregator_type == 'gcn':
                    nf.block_compute(i,
                                    fn.copy_src(src='h', out='m'),
                                    fn.sum('m', 'neigh'),
                                    layer)
                elif self.aggregator_type == 'pool':
                    nf.block_compute(i,
                                    fn.copy_src(src='h', out='m'),
                                    fn.max('m', 'neigh'),
                                    layer)
                elif self.aggregator_type == 'lstm':
                    reducer = self.reducer[i]
                    def _reducer(self, nodes):
                        m = nodes.mailbox['m'] # (B, L, D)
                        batch_size = m.shape[0]
                        h = (m.new_zeros((1, batch_size, self._in_feats)),
                            m.new_zeros((1, batch_size, self._in_feats)))
                        _, (rst, _) = reducer(m, h)
                        return {'neigh': rst.squeeze(0)}

                    nf.block_compute(i,
                                    fn.copy_src(src='h', out='m'),
                                    _reducer,
                                    layer)
                else:
                    raise KeyError('Aggregator type {} not recognized.'.format(self.aggregator_type))
        # set up new feat
        for i in range(lid + 1, nf.num_layers):
            h = nf.layers[i].data.pop('activation')
            nf.layers[i].data['h'] = h

        h = nf.layers[nf.num_layers - 1].data.pop('h')
        return h


class DGLGraphDataParallel(torch.nn.Module):
  """
  Similar to `torch.nn.DataParallel`
  Each element (instance of dgl.NodeFlow) will call 
    `dgl.NodeFlow.copy_from_parent(ctx)`
  to load needed features into corresponding GPUs
  """
  def __init__(self, module, device_ids=None, output_device=None, dim=0):
    super(DGLGraphDataParallel, self).__init__()
    self.use_cuda = True

    if not torch.cuda.is_available():
      self.module = module
      self.device_ids = []
      self.use_cuda = False
      return
    
    if device_ids is None:
      device_ids = list(range(torch.cuda.device_count()))
    if output_device is None:
      output_device = device_ids[0]
    
    self.dim = dim
    self.module = module
    self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    self.output_device = _get_device_index(output_device, True)
    self.src_device_obj = torch.device("cuda:{}".format(self.device_ids[0]))

    _check_balance(self.device_ids)

    if len(self.device_ids) == 1:
      self.module.cuda(device_ids[0])
  

  def forward(self, inputs, **kwargs):
    """
    inputs should be a list of dgl.NodeFlows when multi-gpus is enabled.
    The length of inputs should be equal (or less) to device num.
    Each element in inputs should be an instance of nodeflow
    """
    if not self.device_ids:
      return self.module(*inputs, **kwargs)
    
    for t in chain(self.module.parameters(), self.module.buffers()):
      if t.device != self.src_device_obj:
        raise RuntimeError("module must have its parameters and buffers "
                           "on device {} (device_ids[0]) but found one of "
                           "them on device: {}".format(self.src_device_obj, t.device))

    if not isinstance(inputs, list):
      inputs = [inputs]
    if len(self.device_ids) < len(inputs):
      raise RuntimeError("device num [{}] is not equal to inputs length [{}]"
                         .format(len(self.device_ids), len(inputs)))
    # replicate kwargs
    kwargs = scatter(kwargs, self.device_ids[:len(inputs)], 0)
    if len(self.device_ids) == 1:
      device = torch.device(0) if self.use_cuda else torch.device('cpu')
      inputs[0].copy_from_parent(ctx=device)
      return self.module(inputs[0])
    elif isinstance(inputs[0], NodeFlow):
      # copy inputs from its parent graph (should reside in cuda:0)
      # better way for small graphs to do this is to replica parent features 
      # to all gpus and load from its own gpu
      for device_id in range(len(inputs)):
        device = torch.device(self.device_ids[device_id])
        inputs[device_id].copy_from_parent(ctx=device)
    replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
    outputs = self.parallel_apply(replicas, inputs, kwargs)
    return self.gather(outputs, self.output_device)

  def replicate(self, module, device_ids):
    return replicate(module, device_ids, not torch.is_grad_enabled())

  def parallel_apply(self, replicas, inputs, kwargs):
    return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])
  
  def gather(self, outputs, output_device):
    return gather(outputs, output_device, dim=self.dim)

       
       
