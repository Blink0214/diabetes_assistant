import random
import numpy as np
import torch
import logging
import os

old_factory = logging.getLogRecordFactory()


def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    record.package = os.path.join(*record.pathname.rsplit(os.path.sep, 2)[-2:])
    return record


logging.setLogRecordFactory(record_factory)
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(package)s[line %(lineno)s]: %(message)s",
                    datefmt="%Y/%m/%d %H:%M:%S")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
