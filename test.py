# encoding: utf-8
from __future__ import division
import numpy as np



if __name__ == '__main__':
    # 第一列
    x =np.array( [0.5,0.5])

    # 第二列
    y =np.array( [0,1])

    z=np.array( [1.5,1.5])

    cov=np.array( [[0.3,0.2],[0.2,0.3]])

    print((x-y).dot(np.transpose(cov)).dot(np.transpose(x-y)))
