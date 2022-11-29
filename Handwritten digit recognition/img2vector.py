import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN

"""
函数说明:将32x32的二进制图像转换为1x1024向量
"""


def img2vector(filename):
    # 创建1x1024零向量
    returnVect = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读一行数据
        lineStr = fr.readline()
        # 每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    # 返回转换后的1x1024向量
    return returnVect

testvector = img2vector('testDigits/5_62.txt')
test1 = testvector[0, 0:31]
test2 = testvector[0, 32:63]
print(test1)
print(test2)