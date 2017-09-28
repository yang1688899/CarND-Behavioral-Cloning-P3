
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 15:52:26 2017

@author: yang
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import random

static = {}
lables = [round(-0.9+0.1*i,1) for i in range(20)]
for lable in lables:
    static[lable]=0
lines=[]
with open('../data/driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    next(reader)
    
    for line in reader:
        if float(line[3])>0.0 and float(line[3])<0.1:
            if round(random.random()*0.58):
                lines.append(line)
        elif float(line[3])>-0.2 and float(line[3])<0.2:
            if round(random.random()):
                lines.append(line)
        else:
            lines.append(line)
            
    for line in lines:
        for lable in lables:
            if float(line[3])<lable:
                static[lable]+=1
                break
print (static)       
x = [i for i in static.keys()]
print(x)
y = [i for i in static.values()]
print(y)

x_pos = np.arange(len(x))

plt.figure(figsize=(12,20))
plt.bar(x_pos, y, align='center', alpha=0.4)
plt.xticks(x_pos, x)
    
