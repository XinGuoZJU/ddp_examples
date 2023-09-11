import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data.distributed import DistributedSampler

import argparse
parser = argparse.ArgumentParser()
# 注意这个参数，必须要以这种形式指定，即使代码中不使用。因为 launch 工具默认传递该参数
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()


# 1) 初始化
torch.distributed.init_process_group(backend="nccl") # default: 'env://'
# torch.distributed.init_process_group(backend="nccl", init_method='env://')

input_size = 1
output_size = 2
batch_size = 30
data_size = 120

# 2） 配置每个进程的gpu
local_rank = args.local_rank
rank = torch.distributed.get_rank()

torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size).to('cuda')

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

dataset = RandomDataset(input_size, data_size)
# 3）使用DistributedSampler
rand_loader = DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         sampler=DistributedSampler(dataset))

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        # self.fc = nn.Linear(input_size, output_size)
        self.fc = torch.nn.Parameter(torch.zeros(input_size)) 

    def forward(self, input):
        # output = self.fc(input)
        output = input + self.fc
        print("  In Model: input size", input.size(),
              "output size", output.size())
        return output

model = Model(input_size, output_size)

# 4) 封装之前要把模型移到对应的gpu
model.to(device)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 5) 封装
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)

optimizer = torch.optim.SGD(model.parameters(), lr=1)

for num, data in enumerate(rand_loader):
    if torch.cuda.is_available():
        input_var = data
    else:
        input_var = data

    output = model(input_var)
    loss = output.sum(-1).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(model.module.fc)
    print("Outside: input size", input_var.size(), "output_size", output.size())


