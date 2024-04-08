if __name__ == "__main__":

    import dgl
    import torch as th
    import dgl.function as fn

    dgl.distributed.initialize(ip_config='ip_config.txt')
    th.distributed.init_process_group(backend='nccl')
    g = dgl.distributed.DistGraph('ogbn-products')

    train_nid = dgl.distributed.node_split(g.ndata['train_mask'])
    valid_nid = dgl.distributed.node_split(g.ndata['val_mask'])

    import torch.nn as nn
    import torch.nn.functional as F
    import dgl.nn as dglnn
    import torch.optim as optim
    from time import time

    import os
    local_rank = int(os.environ["LOCAL_RANK"])
    gpu_count = th.cuda.device_count()
    device = th.device("cuda:{}".format(local_rank % gpu_count))

    '''gcn_msg=fn.copy_src(src='h',out='m')
    gcn_reduce=fn.sum(msg='m',out='h')

    class NodeappleModule(nn.Module):
        def __init__(self, in_feats, out_feats, activation):
            super(NodeappleModule, self).__init__()
            self.fc1 = nn.Linear(in_feats, out_feats)
            self.activation = activation

        def forward(self, node):
            h = self.fc1(node.data["h"])
            if self.activation is not None:
                h = self.activation(h)
            return {'h': h}
    class GCN_Layer(nn.Module):
        def __init__(self, in_feats, out_feats, activation):
            super(GCN_Layer, self).__init__()
            self.applynode=NodeappleModule(self, in_feats, out_feats, activation)
        def forward(self,g,feature):
            g.ndata['h']=feature
            g.updateall(gcn_msg,gcn_reduce)
            g.apply_nodes(func=applynode)
            return g.ndata.pop('h')
    class GCN(nn.module):
        def __init__(self, in_feats, out_feats, activation, n_layers):
            pass'''

    class GAT(nn.Module):
        def __init__(self, in_feats, n_hidden, n_classes, n_layers):
            print(n_classes)
            super().__init__()
            self.n_layers = n_layers
            self.n_hidden = n_hidden
            self.n_classes = n_classes
            self.layers = nn.ModuleList()
            self.layers.append(dglnn.GATConv(
                in_feats, n_hidden, 3, allow_zero_in_degree=True))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.GATConv(
                    n_hidden, n_hidden, 3, allow_zero_in_degree=True))
            self.layers.append(dglnn.GATConv(
                n_hidden, n_classes, 3, allow_zero_in_degree=True))

        def forward(self, blocks, x):
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                x = layer(block, x).mean(1)
                if l != self.n_layers - 1:
                    x = F.relu(x)
            return x

    num_hidden = 256
    num_labels = len(th.unique(g.ndata['labels'][0:g.num_nodes()]))
    num_layers = 2
    lr = 0.001

    model = GAT(g.ndata['feat'].shape[1], num_hidden, num_labels, num_layers)

    model.to(device)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = th.nn.parallel.DistributedDataParallel(model)

    sampler = dgl.dataloading.MultiLayerNeighborSampler([25, 10])
    train_dataloader = dgl.dataloading.DistNodeDataLoader(
        g, train_nid, sampler, batch_size=1024,
        shuffle=True, drop_last=False)
    valid_dataloader = dgl.dataloading.DistNodeDataLoader(
        g, valid_nid, sampler, batch_size=1024,
        shuffle=False, drop_last=False)

    import sklearn.metrics
    import numpy as np

    for epoch in range(50):
        ts = time()
        # Loop over the dataloader to sample mini-batches.
        losses = []
        with model.join():
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                # Load the input features as well as output labels
                blocks = [b.to(device) for b in blocks]
                batch_inputs = g.ndata['feat'][input_nodes].to(device)
                batch_labels = g.ndata['labels'][seeds].to(device)

                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)

                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.detach().cpu().numpy())
                optimizer.step()

        # validation
        predictions = []
        labels = []
        with th.no_grad(), model.join():
            for step, (input_nodes, seeds, blocks) in enumerate(valid_dataloader):
                blocks = [b.to(device) for b in blocks]
                inputs = g.ndata['feat'][input_nodes].to(device)

                labels.append(g.ndata['labels'][seeds].numpy())

                predictions.append(
                    model(blocks, inputs).cpu().argmax(1).numpy())

            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        print('Epoch {}: Validation Accuracy {}'.format(epoch, accuracy))
        te = time()
        print('Epoch {} RANK {} : Time Cost {}'.format(epoch, local_rank, te-ts))

    # print("All Work Clear;;")
