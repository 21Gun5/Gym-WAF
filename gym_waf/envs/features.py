#-*- coding:utf-8 –*-
import numpy as np

class Features(object):
    def __init__(self):
        self.dim = 0
        self.name = ""
        self.dtype = np.float32     # 指定数据类型，astype转换时用到

    def byte_histogram(self,str):
        # 由样本字符串，得到对应字符的，ascii码的列表
        bytes = [ord(ch) for ch in list(str)]

        # 构造字符出现次数的array数组（array与py内置的list、tuple均不同)
        h = np.bincount(bytes, minlength=256)   # 指定长度为256；返回为array，其与list类似：list是py内置的，其内元素的类型可不同；而array为numpy所有的，其内元素类型要相同
        
        # array数组拼接，1+256得257维数组
        return np.concatenate([
            [h.sum()],  # 第一个维度，表示字符串长度
            h.astype(self.dtype).flatten() / h.sum(),  # 剩余256维，表字符出现次数平均值（除长度得）；astype转换为指定数据类型；flatten将多维将为一维
        ])

    def extract(self, str):
        # 返回最终的特征向量
        # shape为(1,257)，可想象1个大元素中有257个小元素，即[ [1, 2, .., 257] ]
        featurevectors = [
            [self.byte_histogram(str)]
        ]
        return np.concatenate(featurevectors)

if __name__ == '__main__':
    f = Features()
    a = f.extract("alert()")
    print(a)
    print(a.shape)
