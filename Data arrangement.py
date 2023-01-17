import numpy as np
import csv

pattern = list()
fhd = csv.reader(open('输入3.csv', 'r', encoding='utf-8-sig'))

for line in fhd:
    for i in range(len(line)):
        line[i] = float(line[i])
    line = np.array(line)
    pattern.append(line)

for i in range(0, 735, 15):
    number11 = pattern[i+10]
    number12 = pattern[i+11]
    number13 = pattern[i+12]
    number14 = pattern[i+13]
    number15 = pattern[i+14]
    pattern[i+5], pattern[i+8] = pattern[i+8], pattern[i+5]
    pattern[i+10], pattern[i+11] = number14, number15
    pattern[i+12], pattern[i+13], pattern[i+14] = number11, number12, number13

output = csv.writer(open('输入4.csv', 'a', newline=''), dialect='excel')
output.writerows(map(lambda x: x, pattern))