if __name__=="__main__":

    import dgl
    import torch as th
    import csv

    dgl.distributed.initialize(ip_config='ip_config.txt')
    th.distributed.init_process_group(backend='nccl')
    g = dgl.distributed.DistGraph('ogbn-arxiv')


    train_nid = dgl.distributed.node_split(g.ndata['train_mask'])
    valid_nid = dgl.distributed.node_split(g.ndata['val_mask'])


    import torch.nn as nn
    import torch.nn.functional as F
    import dgl.nn as dglnn
    import torch.optim as optim

    import os
    global_rank = int(os.environ["RANK"])
    from time import time
    local_rank = int(os.environ["LOCAL_RANK"])
    gpu_count=th.cuda.device_count()
    device = th.device("cuda:{}".format(local_rank % gpu_count))

    class SAGE(nn.Module):
        def __init__(self, in_feats, n_hidden, n_classes, n_layers):
            super().__init__()
            self.n_layers = n_layers
            self.n_hidden = n_hidden
            self.n_classes = n_classes
            self.layers = nn.ModuleList()
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))

        def forward(self, blocks, x):
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                x = layer(block, x)
                if l != self.n_layers - 1:
                    x = F.relu(x)
            return x

    num_hidden = 256
    num_labels = len(th.unique(g.ndata['labels'][0:g.num_nodes()]))
    num_layers = 2
    lr = 0.001


    model = SAGE(g.ndata['feat'].shape[1], num_hidden, num_labels, num_layers)

    model.to(device)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = th.nn.parallel.DistributedDataParallel(model)


    sampler = dgl.dataloading.MultiLayerNeighborSampler([25,10])
    train_dataloader = dgl.dataloading.DistNodeDataLoader(
                                    g, train_nid, sampler, batch_size=1024,
                                    shuffle=True, drop_last=False)
    valid_dataloader = dgl.dataloading.DistNodeDataLoader(
                                    g, valid_nid, sampler, batch_size=1024,
                                    shuffle=False, drop_last=False)

    import sklearn.metrics
    import numpy as np

    for epoch in range(200):
        tepoch_start=time()
        # Loop over the dataloader to sample mini-batches.
        tdataloadArray=[]
        ttrainArray=[]
        losses = []
        with model.join():
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                # Load the input features as well as output labels
                tdataload_start=time()
                blocks = [b.to(device) for b in blocks]
                batch_inputs = g.ndata['feat'][input_nodes].to(device)
                batch_labels = g.ndata['labels'][seeds].to(device)
                ttrain_start=time()
                tdataloadArray.append(ttrain_start-tdataload_start)
                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.detach().cpu().numpy())
                tbatch_end=time()
                ttrainArray.append(tbatch_end-ttrain_start)
                optimizer.step()
        tepoch_end=time()

        # validation
        predictions = []
        labels = []
        with th.no_grad(), model.join():
            for step, (input_nodes, seeds, blocks) in enumerate(valid_dataloader):
                blocks = [b.to(device) for b in blocks]
                inputs = g.ndata['feat'][input_nodes].to(device)

                labels.append(g.ndata['labels'][seeds].numpy())

                predictions.append(model(blocks, inputs).cpu().argmax(1).numpy())

            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            accuracy = sklearn.metrics.accuracy_score(labels, predictions)

        print('Epoch {0} RANK {1} : acc {2} dataload {3} train {4} epoch {5}'.format(epoch, global_rank, accuracy,sum(tdataloadArray),sum(ttrainArray),tepoch_end-tepoch_start))
        with open('acc_asge_ar{}.csv'.format(global_rank), mode='a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, global_rank, accuracy,sum(tdataloadArray),sum(ttrainArray),tepoch_end-tepoch_start])
    

    #print("All Work Clear;;")