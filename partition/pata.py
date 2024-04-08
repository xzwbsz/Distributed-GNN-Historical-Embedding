import os
os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import torch as th
from ogb.nodeproppred import DglNodePropPredDataset
data = DglNodePropPredDataset(name='Reddit')
graph, labels = data[0]
labels = labels[:, 0]
graph.ndata['labels'] = labels

splitted_idx = data.get_idx_split()
train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
train_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
train_mask[train_nid] = True
val_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
val_mask[val_nid] = True
test_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
test_mask[test_nid] = True
graph.ndata['train_mask'] = train_mask
graph.ndata['val_mask'] = val_mask
graph.ndata['test_mask'] = test_mask

'''dgl.distributed.partition_graph(graph, graph_name='ogbn-products', num_parts=4,
                            out_path='4part_data',
                            balance_ntypes=graph.ndata['train_mask'],
                            balance_edges=True)'''

nmap, emap = dgl.distributed.partition_graph(graph, graph_name='ogbn-arxiv',
                                            num_parts=2,
                                            out_path='Reddit2part_data',
                                            balance_ntypes=graph.ndata['train_mask'],
                                            balance_edges=True,
                                            return_mapping=True)
orig_node_emb = th.zeros(node_emb.shape, dtype=node_emb.dtype)
orig_node_emb[nmap] = node_emb