import numpy as np
# 数据可视化
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import *
import operator
from file2matrix import *
from autoNorm import *

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet      # 距离计算
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    # 选择距离最小的K个点 把classCount分解为元组列表
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        # itemgetter为排序 为逆序，即按照最大到最小的次序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#测试算法：作为完整程序验证分类器
def datingClassTest():
        filename = "datingTestSet.txt"
        datingDataMat, datingLabels = file2matrix(filename)
        hoRatio = 0.10
        normMat, ranges, minVals = autoNorm(datingDataMat)
        m = normMat.shape[0]
        numTestVecs = int(m * hoRatio)
        errorCount = 0.0
        for i in range(numTestVecs):
            classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                         datingLabels[numTestVecs:m], 4)
            print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
            if classifierResult != datingLabels[i]:
                errorCount += 1.0
        print("错误率:%f%%" % (errorCount / float(numTestVecs) * 100))


