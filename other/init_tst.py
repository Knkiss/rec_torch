import random, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


class dataset(Dataset):
    def __init__(self):
        self.data = np.linspace(1,100,10,dtype='int')
        self.size = self.data.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    """
    pytorch的init次数会导致shuffle的结果不一致，使得性能表现不公平
    修改double_init 观察打印结果变化
    """
    double_init = False

    set_seed(2020)
    embA = nn.Parameter(nn.init.normal_(torch.empty(10, 3), std=0.1))
    if double_init:
        embA = nn.Parameter(nn.init.normal_(torch.empty(10, 3), std=0.1))

    data = dataset()
    UILoader = DataLoader(data, batch_size=20, shuffle=True, num_workers=0)
    for i in UILoader:
        print(i)

