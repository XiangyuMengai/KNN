import numpy as np
# 数据可视化
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import *
import operator

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)                #得到文本行数
    returnMat = np.zeros((numberOfLines,3))      #创建返回的NumPy二维矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        # line.split()截取掉所有回车字符 用tab字符\t将上一步得到的整行数据分割成一个元素列表
        listFromLine = line.split('\t')
        # 选取前面三个元素，存储到特征矩阵中
        returnMat[index,:] = listFromLine[0:3]
        # 使用索引值-1为表示列表中最后一列元素
        if listFromLine[-1] == 'didntLike':      #文本内收集的数据
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector
