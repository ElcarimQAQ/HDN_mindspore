import mindspore as ms
from  mindspore import ops
import numpy as np

if __name__ == '__main__':
    pred_cls = ms.Tensor(np.load('/home/lbyang/workspace/HDN_mindspore/pred_cls.npy'))
    neg = ms.Tensor(np.load('/home/lbyang/workspace/HDN_mindspore/neg.npy'))
    for i in range(100):
        print("The { %d } Time", i)
        print(ops.gather(pred_cls, neg, 0)[:, 1])
