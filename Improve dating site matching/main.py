from file2matrix import *
from showdatas import *
from autoNorm import *
from datingClassTest import *
from classifyPerson import *

if __name__ == '__main__':
    #打开的文件名
    filename = "datingTestSet.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    print(datingDataMat)
    print(datingLabels)
    showdatas(datingDataMat, datingLabels)

    # 归一化特征值
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    print(normDataSet)
    print(ranges)
    print(minVals)

    # 测试算法
    datingClassTest()

    # 约会网站预测函数
    classifyPerson()