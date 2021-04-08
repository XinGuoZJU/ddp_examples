# https://zhuanlan.zhihu.com/p/76638962
# https://zhuanlan.zhihu.com/p/74792767


# tcp; multi-device nulti-gpu
# python torch_tcp.py --init_method tcp://10.76.0.143:10086 --rank 0 --world_size 2 --local_rank 0

# env: multi-device nulti-gpu 
# python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="10.76.0.143" --master_port=10086 torch_env.py --local_rank 0

# env: single-device multi-gpu
# python -m torch.distributed.launch --nproc_per_node=2 torch_env.py


# test batch_size
# python -m torch.distributed.launch --nproc_per_node=1 torch_env_test.py
# python -m torch.distributed.launch --nproc_per_node=2 torch_env_test.py

