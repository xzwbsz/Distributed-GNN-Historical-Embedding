/home/gnn/miniconda3/envs/dgl/bin/python launch.py \
  --workspace ./ \
  --num_trainers 2 \
  --num_samplers 1 \
  --num_servers 2 \
  --part_config part_data/ogbn-arxiv.json \
  --ip_config ip_config.txt \
  "/nn/fyJu.py"