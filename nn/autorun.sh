/home/gnn/miniconda3/envs/dgl/bin/python /home/gnn/workplace/hanjzTEST/dglexp/launch.py \
  --workspace ~/workplace/hanjzTEST/dglexp/ \
  --num_trainers 2 \
  --num_samplers 1 \
  --num_servers 2 \
  --part_config ogbn_arxiv2part_data/ogbn-arxiv.json \
  --ip_config ip_config.txt \
  "/home/gnn/miniconda3/envs/dgl/bin/python /home/gnn/workplace/hanjzTEST/dglexp/nn/gat_o.py"