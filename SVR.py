import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
import sys
import csv
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore") # 忽略警告

# 代码说明：需要准备一个原始光谱数据文件pattern.csv，原始label文件label.csv，代码最后会输出一个PCA结果文件pattern_pca.csv以及SVR预测结果文件label_predict.pca

n = 0

names = ["有效碳氢比", "热值", "不饱和浓度", "溶液的氢质量分数", "溶液的氧质量分数", "溶液的碳质量分数"]

name = "输出"
#for name in names:
#for n in range(1, 21):
#for b in range(0, n):
# 读取光谱数据到list变量pattern
pattern = list()
fhd = csv.reader(open('输入4.csv', 'r', encoding='utf-8-sig'))
for line in fhd:
    pattern.append(line)
'''
# 对光谱数据进行主成分分析
pattern = np.array(pattern, dtype='float64')  # 调整数组大小。
pca = PCA(n_components=n)  # n_components选择主成分个数
pca.fit(pattern)  # fit()可以说是scikit-learn中通用的方法，每个需要训练的算法都会有fit()方法，它其实就是算法中的“训练”这一步骤。
pattern = pca.transform(pattern)  # 将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。
pattern = pattern.tolist()  # 列表化

#b = 0

new_data = []
for row in pattern:
    list1 = []
    for c in range(0, n):
        #row[b] = float(row[b]) * 1.3
        if c != b :
            list1.append(row[c])
    new_data.append(list1)

#print(new_data)
# 将主成分分析结果写入新的CSV文件
output = csv.writer(open('pattern_pca%d_pc%d.csv'%(n,b+1),'a',newline=''),dialect='excel')
output.writerows(new_data)
'''
# 读取label数据到list变量label
label = list()
fhl = csv.reader(open('%s.csv'%name,'r',encoding='utf-8-sig'))
for line in fhl:
    label.append(line)
label = np.array(label,dtype = 'float64')
label = label.tolist()

a = 4
#for a in range(1, 5):
# 进行SVR数据准备
raw_train = list()
label_train = list()
raw_test = list()
label_test = list()
raw_cross = list()
label_cross = list()
i, j = 1, 1
for line in pattern:
    if i == a:
        raw_cross.append(line)
        i = i + 1
    elif i == 5:
        raw_test.append(line)
        i = 1
    else:
        raw_train.append(line)
        i = i + 1
for line in label:
    if j == a:
        label_cross.append(line)
        j = j + 1
    elif j == 5:
        label_test.append(line)
        j = 1
    else:
        label_train.append(line)
        j = j + 1

# 进行SVR并将预测结果写入新的CSV文件
mods = ['linear','rbf','poly']
for mod in mods:
    clf = svm.SVR(kernel = mod)  # kernel变量分别为linear、rbf、poly
    clf.fit(raw_train, np.ravel(label_train))  # 注意label_train的转置
    label_predict = clf.predict(raw_cross)
    label_predict = label_predict.tolist()
    for i in range(len(label_predict)):
        for j in range(i + 1, len(label_predict)):
            if label_predict[i] > label_predict[j]:
                num = label_predict[i]
                label_predict[i] = label_predict[j]
                label_predict[j] = num
    output = csv.writer(open('label_cross_predict_%s_%s_pca%d_交叉训练集%d.csv' % (name, mod, n, a), 'a', newline=''), dialect='excel')
    output.writerows(map(lambda x: [x], label_predict))

    label_a = []

    a = 0

    for i in range(len(label_predict)):
        a = a + label_predict[i]

    label_a.append(a)

    b = 0

    for i in range(len(label_cross)):
        b = b + label_cross[i][0]

    label_a.append(b)

    c = 1 - abs(b - a) / b
    label_a.append(c)

    output = csv.writer(
        open('累加准确率_label_cross_predict_%s_%s_pca%d_交叉训练集%d.csv' % (name, mod, n, a), 'a', newline=''),
        dialect='excel')
    output.writerows(map(lambda x: [x], label_a))

    clf = svm.SVR(kernel = mod) # kernel变量分别为linear、rbf、poly
    clf.fit(raw_train, np.ravel(label_train)) # 注意label_train的转置
    label_predict = clf.predict(raw_test)
    label_predict = label_predict.tolist()
    for i in range(len(label_predict)):
        for j in range(i + 1, len(label_predict)):
            if label_predict[i] > label_predict[j]:
                num = label_predict[i]
                label_predict[i] = label_predict[j]
                label_predict[j] = num
    output = csv.writer(open('label_predict_%s_%s_pca%d_交叉训练集%d.csv'%(name, mod, n, a) ,'a',newline = ''),dialect = 'excel')
    output.writerows(map(lambda x:[x],label_predict))

    label_a = []

    a = 0

    for i in range(len(label_predict)):
        a = a + label_predict[i]

    label_a.append(a)

    b = 0

    for i in range(len(label_cross)):
        b = b + label_cross[i][0]

    label_a.append(b)

    c = 1 - abs(b - a) / b
    label_a.append(c)

    output = csv.writer(
        open('累加准确率_label_predict_%s_%s_pca%d_交叉训练集%d.csv' % (name, mod, n, a), 'a', newline=''),
        dialect='excel')
    output.writerows(map(lambda x: [x], label_a))

output = csv.writer(open('label_cross_%s_pca%d_交叉训练集%d.csv' % (name, n, a), 'a', newline=''), dialect='excel')
output.writerows(map(lambda x: x, label_cross))
output = csv.writer(open('label_test_%s_pca%d_交叉训练集%d.csv' % (name, n, a), 'a', newline=''), dialect='excel')
output.writerows(map(lambda x: x, label_test))
