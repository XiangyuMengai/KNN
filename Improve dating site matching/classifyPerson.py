import numpy as np
# 数据可视化
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import *
import operator
from file2matrix import *
from autoNorm import *
from datingClassTest import *

# 约会网站预测函数
def classifyPerson():
        resultList = ['讨厌','有些喜欢','非常喜欢']
        precentTats = float(input("玩视频游戏所耗时间百分比:"))
        ffMiles = float(input("每年获得的飞行常客里程数:"))
        iceCream = float(input("每周消费的冰激淋公升数:"))
        filename = "datingTestSet.txt"
        datingDataMat, datingLabels = file2matrix(filename)
        normMat, ranges, minVals = autoNorm(datingDataMat)
        inArr = np.array([precentTats, ffMiles, iceCream])
        norminArr = (inArr - minVals) / ranges
        classifierResult = classify0(norminArr, normMat, datingLabels, 3)
        print("你可能%s这个人" % (resultList[classifierResult - 1]))


