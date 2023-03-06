import torch
flag = torch.cuda.is_available()
print(flag)

n_gpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")
print(device)

if flag:
    print(torch.cuda.get_device_name(0))
    print(torch.rand(3, 3).cuda())
else:
    print(torch.rand(3, 3))
