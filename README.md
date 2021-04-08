# ddp_examples
PyTorch Data Distributed Parallel examples

references:

  https://zhuanlan.zhihu.com/p/76638962
  
  https://zhuanlan.zhihu.com/p/74792767


Test Batch_size: In our setting, the lr is 1.0, and the gradient is always 1.0.

Results:

    batch_size = 15, 1 GPU, loss sum:
    -15, -30, -45, -60, -75, -90, -105, -120
    batch_size = 15, 2 GPU, loss sum:
    -15, -30, -45, -60

    batch_size = 30, 1 GPU, loss sum:
    -30, -60, -90, -120
    batch_size = 30, 2 GPU, loss sum:
    -30, -60


    batch_size = 15, 1 GPU, loss mean:
    -1, -2, -3, -4, -5, -6, -7, -8
    batch_size = 15, 2 GPU, loss mean:
    -1, -2, -3, -4

    batch_size = 30, 1 GPU, loss mean:
    -1, -2, -3, -4
    batch_size = 30, 2 GPU, loss mean:
    -1, -2


https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel

When a model is trained on M nodes with batch=N, the gradient will be M times smaller when compared to the same model trained on a single node with batch=M*N if the loss is summed (NOT averaged as usual) across instances in a batch (because the gradients between different nodes are averaged). You should take this into consideration when you want to obtain a mathematically equivalent training process compared to the local training counterpart. But in most cases, you can just treat a DistributedDataParallel wrapped model, a DataParallel wrapped model and an ordinary model on a single GPU as the same (E.g. using the same learning rate for equivalent batch size).
