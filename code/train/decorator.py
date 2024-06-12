import time
import torch

from torch.autograd import profiler


def count_time_once(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print("函数", func.__name__, "的执行时间为：", execution_time, "秒")
        return result
    return wrapper


def count_cpu_once(func):
    def wrapper(*args, **kwargs):
        with profiler.profile(record_shapes=True, use_cuda=torch.cuda.is_available()) as prof:
            with profiler.record_function("model_training"):
                func(*args, **kwargs)
        print("函数", func.__name__, "的性能统计如下：")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        exit()
    return wrapper
