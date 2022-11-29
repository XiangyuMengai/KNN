import numpy as np
# 数据可视化
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import *
import operator

# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)         #从列中选取最小值
    maxVals = dataSet.max(0)         #从列中选取最大值
    # 当前值减去最小值，然后除以取值范围
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))     #特征值相除
    return normDataSet, ranges, minVals

