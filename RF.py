import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
import csv
from sklearn.neural_network import MLPRegressor
import warnings

warnings.filterwarnings("ignore") # 忽略警告

# 代码说明：需要准备一个原始光谱数据文件pattern.csv，原始label文件label.csv，代码最后会输出一个PCA结果文件pattern_pca.csv以及SVR预测结果文件label_predict.pca

n = 0

names = ["有效碳氢比", "热值", "不饱和浓度", "溶液的氢质量分数", "溶液的氧质量分数", "溶液的碳质量分数"]

name = "输出"
#for name in names:
mods = ['lbfgs']

for mod in mods:
    TN = []
    FP = []
    FN = []
    TP = []
    ALL = []
# for name in names:
    moods = ['identity']
    for mood in moods:
        #for n in range(1, 8):
        # for b in range(0, n):
        # 读取光谱数据到list变量pattern
        pattern = list()
        fhd = csv.reader(open('输入4.csv', 'r', encoding='utf-8-sig'))
        for line in fhd:
            pattern.append(line)
        pattern = np.array(pattern, dtype='float64')
        pattern = pattern.tolist()
        '''
        # 对光谱数据进行主成分分析
        pattern = np.array(pattern, dtype='float64')  # 调整数组大小。
        pca = PCA(n_components=n)  # n_components选择主成分个数
        pca.fit(pattern)  # fit()可以说是scikit-learn中通用的方法，每个需要训练的算法都会有fit()方法，它其实就是算法中的“训练”这一步骤。
        pattern = pca.transform(pattern)  # 将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。
        pattern = pattern.tolist()  # 列表化

        # b = 0
        
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
        fhl = csv.reader(open('%s.csv' % name, 'r', encoding='utf-8-sig'))
        for line in fhl:
            label.append(line)
        label = np.array(label, dtype='float64')
        label = label.tolist()

        a = 4
        # for a in range(1, 5):
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
        # clf = svm.SVC(kernel = mod)  # 分类模型
        # clf = svm.SVR(kernel=mod)  # 回归模型
        clf = MLPRegressor(solver=mod, activation=mood, random_state=1)
        clf.fit(raw_train, np.ravel(label_train))  # 注意label_train的转置
        label_predict = clf.predict(raw_cross)
        label_predict = label_predict.tolist()

        for i in range(len(label_predict)):
            for j in range(i + 1, len(label_predict)):
                if label_predict[i] > label_predict[j]:
                    num = label_predict[i]
                    label_predict[i] = label_predict[j]
                    label_predict[j] = num

        # 交叉验证集
        output = csv.writer(open('label_cross_predict_%s_%s_%s_pca%d.csv' % (name, mod, mood, n), 'a', newline=''), dialect='excel')
        output.writerows(map(lambda x: [x], label_predict))


        # 测试集
        label_predict = clf.predict(raw_test)
        label_predict = label_predict.tolist()

        for i in range(len(label_predict)):
            for j in range(i + 1, len(label_predict)):
                if label_predict[i] > label_predict[j]:
                    num = label_predict[i]
                    label_predict[i] = label_predict[j]
                    label_predict[j] = num

        output = csv.writer(open('label_test_predict_%s_%s_%s_pca%d.csv' % (name, mod, mood, n), 'a', newline=''), dialect='excel')
        output.writerows(map(lambda x: [x], label_predict))

        '''
        a,b,c,d = 0,0,0,0
        
        for i in range(len(label_predict)):
            if label_predict[i] == label_cross[i][0] == 0:
                a = a + 1
            elif (label_predict[i] == 1) and (label_cross[i][0] == 0):
                b = b + 1
            elif (label_predict[i] == 0) and (label_cross[i][0] == 1):
                c = c + 1
            elif label_predict[i] == label_cross[i][0] == 1:
                d = d + 1
        
        TN.append(a)
        FP.append(b)
        FN.append(c)
        TP.append(d)
        
        label_a = []
        
        for i in range(len(label_predict)):
            accuracy = (1 - np.abs((label_cross[i][0] - label_predict[i]) / label_cross[i][0]))*100
            label_a.append(accuracy)
        
        output = csv.writer(open('准确率_label_predict_%s_%s_pca%d.csv' % (name, mod, n), 'a', newline=''), dialect='excel')
        output.writerows(map(lambda x: [x], label_a))
        
        
        #pca缩进线
        output = csv.writer(open('分类label_cross_predict_%s_%s.csv' %(name, mod), 'a', newline=''),
                    dialect='excel')
        output.writerow(TN)
        output.writerow(FP)
        output.writerow(FN)
        output.writerow(TP)
        
       
        '''
        output = csv.writer(open('label_cross_%s_%s_%s_pca%d.csv' % (name, mod, mood, n), 'a', newline=''), dialect='excel')
        output.writerows(map(lambda x: x, label_cross))
        output = csv.writer(open('label_test_%s_%s_%s_pca%d.csv' % (name, mod, mood, n), 'a', newline=''), dialect='excel')
        output.writerows(map(lambda x: x, label_test))
