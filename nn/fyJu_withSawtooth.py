if __name__ == "__main__":

    import dgl
    import torch as th
    import dgl.function as fn
    import csv
    import random

    dgl.distributed.initialize(ip_config='ip_config.txt')
    th.distributed.init_process_group(backend='nccl')
    g = dgl.distributed.DistGraph('ogbn-products')
    # print(g.__dict__)

    train_nid = dgl.distributed.node_split(g.ndata['train_mask'])
    # trainNid = train_nid.tolist()
    # print('tr',trainNid)
    valid_nid = dgl.distributed.node_split(g.ndata['val_mask'])
    # print('gshape',g.shape)
    # print('g',g.__dir__)
    # print('train',train_nid)
    # g.ndata['hist1']=g.ndata['feat']
    # g.ndata['hist1'][0] = g.ndata['hist1'][1]
    # g.ndata['train_list']=th.zeros
    # print('hist1_after',g.ndata['hist1'][0],g.ndata['hist1'][1])
    # print('feat_beofre',g.ndata['feat'][0],g.ndata['feat'][1])

    import torch.nn as nn
    import torch.nn.functional as F
    import dgl.nn as dglnn
    import torch.optim as optim
    from time import time
    from modules.AME import AME
    from modules.Sift import Sift

    import os
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    gpu_count = th.cuda.device_count()
    device = th.device("cuda:{}".format(local_rank % gpu_count))
    bs = 1024

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
    def f_r_select(rn, vb=1):
        def rte(func):
            def inner(*args, **kwargs):
                if (global_rank == rn):
                    if vb:
                        print("Rank_{} ".format(rn), end='')
                    func(*args, **kwargs)
            return inner
        return rte

# find var in 2nd floor+ stack mem
    import inspect

    def retrieve_name_onevar(var):
        for fi in inspect.stack()[1:]:
            for item in fi.frame.f_locals.items():
                if var is item[1]:
                    return item[0]
        return ""
    vn1 = retrieve_name_onevar

    def cut_slt(epoch, batch, l, fl):
        if not fl:
            return False
        # return epoch%3!=0
        # rprint("cut_slt")
        return True

    def update_slt(epoch, batch, l, fl):
        if not fl:
            return False
        # return epoch%3!=0
        # rprint("update_slt")
        return True

    def cut_num_select(epoch, batch, l, gidx, xidx):
        if l > 1:
            return False
        if xidx < bs:
            return False
        try:
            rwres = dgl.sampling.random_walk(g._g, gidx.cpu(), length=5)[0][0]
            # rprint(rwres)
            for _ in rwres:
                if _ in seeds:
                    # print("Atatta!")
                    return True
        except:
            pass
        try:
            if (hist[l][gidx.item()][1]-epoch<=3):
                #rprint(weightlist[hist[l][gidx.item()][1]])
                return True
        except KeyError:
            return False
        # if _ in seeds:
        # return True
        # return random.randint(0,2)!=0
        # rprint("cut_num_select")
        return False

    def update_num_select(epoch, batch, l, gidx, xidx):
        # return random.randint(0, 1) == 0
        return True
        if (xidx < bs and l == 3):
            return True
        # if(l<2 and random.randint(0,2)==1):
        if (l < 2 and random.randint(0, 4) != 1):
            # print(xidx)
            return True
        # rprint("!update_num_select")
        return False

    class GCN(nn.Module):
        def __init__(self, in_feats, n_hidden, n_classes, n_layers):
            super().__init__()
            self.n_layers = n_layers
            self.n_hidden = n_hidden
            self.n_classes = n_classes
            self.layers = nn.ModuleList()
            self.layers.append(dglnn.GraphConv(
                in_feats, n_hidden, allow_zero_in_degree=True))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.GraphConv(
                    n_hidden, n_hidden, allow_zero_in_degree=True))
            self.layers.append(dglnn.GraphConv(
                n_hidden, n_classes, allow_zero_in_degree=True))

        # def forward(self, blocks, x, epoch):
        #     #print(x.shape)
        #     if (epoch % 3 == 0):
        #         for l, (layer, block) in enumerate(zip(self.layers, blocks)):
        #             x = layer(block, x)
        #             if l != self.n_layers - 1:
        #                 x = F.relu(x)
        #         np.save("x1_{}.npy".format(local_rank),
        #                 x.detach().cpu())
        #     else:
        #         x = th.tensor(
        #             np.load("x1_{}.npy".format(local_rank))).to(device)
        #     return x

        def forward(self, blocks, x, epoch=0, batch=0, fl=0):
            if epoch == 0 and batch == 0:
                for _ in range(len(self.layers)):
                    hist.append({})
            if (batch > 0):
                for _ in str("Rank_{} ".format(global_rank)+str(batch-1)):
                    rvbprint("\b", end="")
                rvbprint(batch, end=" ")

            for l, (layer, block) in enumerate(zip(self.layers, blocks)):

                # rprint(l)
                # Cut CalcG
                x = layer(block, x)
                if l != self.n_layers - 1:
                    x = F.relu(x)
                # print("xshape",x.shape)

                tcs = time()
                if cut_slt(epoch, batch, l, fl):
                    for xidx, gidx in enumerate(block._node_frames[1]['_ID']):
                        # rprint(epoch,batch,l,gidx,xidx)
                        # rprint(hist[l])
                        try:
                            if cut_num_select(epoch, batch, l, gidx, xidx):
                                x[xidx].data = hist[l][gidx.item()][0].to(
                                    device).data
                                # print(gidx)
                        except KeyError:
                            pass
                tce = time()
                global tc
                tc += tce-tcs

                tus = time()
                # rprint("j")
                if update_slt(epoch, batch, l, fl):
                    for xidx, gidx in enumerate(block._node_frames[1]['_ID']):
                        if update_num_select(epoch, batch, l, gidx, xidx):
                            hist[l][gidx.item()] = [x[xidx].cpu(),max(epoch-1,0),0]
                    # rprint(blocks[-1]._node_frames[1]['_ID'].shape)
                    # rprint("r")
                tue = time()
                global tu
                tu += tue-tus
            return x

    sample_n_num = [25, 10]
    num_hidden = 256
    num_labels = len(th.unique(g.ndata['labels'][0:g.num_nodes()]))

    # fix block!
    num_layers = len(sample_n_num)

    lr = 0.001

    model = GCN(g.ndata['feat'].shape[1], num_hidden, num_labels, num_layers)

    model.to(device)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # model = th.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
    model = th.nn.parallel.DistributedDataParallel(model)
    from dgl.dataloading import BlockSampler

    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_n_num)
    train_dataloader = dgl.dataloading.DistNodeDataLoader(
        g, train_nid, sampler, batch_size=bs,
        shuffle=True, drop_last=False)
    valid_dataloader = dgl.dataloading.DistNodeDataLoader(
        g, valid_nid, sampler, batch_size=bs,
        shuffle=False, drop_last=False)

    @ f_r_select(0)
    def rprint(*args, **kwargs):
        return print(*args, **kwargs)

    @ f_r_select(0, 0)
    def rvbprint(*args, **kwargs):
        return print(*args, **kwargs)

    import sklearn.metrics
    import numpy as np

    hist = []
    #rprint(model.module.layers.__dict__)
    weightlist = [th.zeros(model.module.layers[0].weight.shape)]
    for epoch in range(50):
        ts = time()
        losses = []
        # Main Loop
        with model.join():
            tc, tu = 0.0, 0.0
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                # Data Load
                blocks = [b.to(device) for b in blocks]
                batch_inputs = g.ndata['feat'][input_nodes].to(device)
                batch_labels = g.ndata['labels'][seeds].to(device)
                # rprint(seeds.shape)

                # forward
                batch_pred = model(blocks, batch_inputs, epoch, step, fl=1)
                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.detach().cpu().numpy())
                optimizer.step()
        te = time()
        weightlist.append(model.module.layers[0].weight)

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

        print('Epoch {0} RANK {1} : Validation Accuracy {2} {3} {4} {5}'.format(
            epoch, global_rank, accuracy, te-ts, tc, tu))
        # print('csv saved')

        # with open(r'./gcnres/2Laa_acc_r{}.csv'.format(global_rank), mode='a+', newline='') as file:
        with open(r'./testr{}.csv'.format(global_rank), mode='a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, accuracy, te-ts, tc, tu])

    # print("All Work Clear;;")
