import pandas as pd
import csv

df = pd.read_csv('数据待增强.csv', header=None, engine='python')

L = 1
q = 7
m = 3
data = list()

for i in range(0, 249):
    for j in range(1, m+1):
        temp = list()
        temp = df[i][L * (j - 1):L * (j - 1) + q]
        # print(temp)
        data.append(temp)

for i in range(len(data)):
    f = open("输入3.csv", 'a', newline='', encoding='utf-8-sig')
    writer = csv.writer(f)
    writer.writerow(data[i])
    f.close()
