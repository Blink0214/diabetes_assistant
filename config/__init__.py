import random
import numpy as np
import torch
import logging
import os

# 保存旧的日志记录工厂函数
old_factory = logging.getLogRecordFactory()

# 定义新的日志记录工厂函数
def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    # 获取文件路径的最后两级目录
    record.package = os.path.join(*record.pathname.rsplit(os.path.sep, 2)[-2:])
    return record

# 设置新的日志记录工厂函数
logging.setLogRecordFactory(record_factory)

# 配置基本日志记录
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(package)s[line %(lineno)s]: %(message)s",
                    datefmt="%Y/%m/%d %H:%M:%S")


def setup_seed(seed):
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed) # 设置 CPU 上的随机种子。
    torch.cuda.manual_seed_all(seed) # 设置所有 GPU 上的随机种子
    torch.cuda.manual_seed(seed) # 冗余设置，确保所有 GPU 的随机数生成器都使用相同的种子。
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    # 设置 Python 标准库的随机种子
    random.seed(seed)
    # 确保使用确定性算法进行卷积运算
    torch.backends.cudnn.deterministic = True
