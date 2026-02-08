import pandas as pd
import pickle as pk
import os
import csv
import random
from sklearn.metrics import accuracy_score as acs
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

frame_number = 3
train_data = []
train_label = []

for file in os.listdir('../trainset'):
    current_file = []
    if(not 'csv' in file):
        continue
    f = csv.reader(open('../trainset/' + file))
    flag = 0 #skip 1st line
    for line in f:
        if(flag == 0):
            flag = 1
            continue
        current_file.append(line)
    length = len(current_file)

    for i in range(0, length):
        for j in range(0, len(current_file[i])):
            current_file[i][j] = float(current_file[i][j])
    
    for i in range(frame_number-1, length):
        frames = []
        for j in range(i+1-frame_number, i+1):
            frame = current_file[j]
            for x in range(9, 75, 3):
                frame[x] -= current_file[j][0]
            for y in range(10, 75, 3):
                frame[y] -= current_file[j][1]
            for z in range(11, 75, 3):
                frame[z] -= current_file[j][2]
            frames = frames + frame[9:-2]
        train_data.append(frames)
        train_label.append(current_file[i][-2])

random.seed(0)
random.shuffle(train_data)
random.seed(0)
random.shuffle(train_label)

test_data = []
test_label = []

for file in os.listdir('../testset'):
    current_file = []
    if(not 'csv' in file):
        continue
    f = csv.reader(open('../testset/' + file))
    flag = 0
    for line in f:
        if(flag == 0):
            flag = 1
            continue
        current_file.append(line)
    length = len(current_file)

    for i in range(0, length):
        for j in range(0, len(current_file[i])):
            current_file[i][j] = float(current_file[i][j])
    
    for i in range(frame_number-1, length):
        frames = []
        for j in range(i+1-frame_number, i+1):
            frame = current_file[j]
            for x in range(9, 75, 3):
                frame[x] -= current_file[j][0]
            for y in range(10, 75, 3):
                frame[y] -= current_file[j][1]
            for z in range(11, 75, 3):
                frame[z] -= current_file[j][2]
            frames = frames + frame[9:-2]
        test_data.append(frames)
        test_label.append(current_file[i][-2])

random.seed(0)
random.shuffle(test_data)
random.seed(0)
random.shuffle(test_label)

# training and saving model
# MLP
mlp = MLPClassifier(hidden_layer_sizes=(50, 100, 50), max_iter=500, random_state=0)
mlp.fit(train_data, train_label)
pre = pd.DataFrame(mlp.predict(test_data))
accu = acs(test_label, pre)
print('MLP ' + 'accuracy: {:.2%}'.format(accu))
"""with open('../model/mlp_gesture.pickle', 'wb')as f:
    pk.dump(mlp, f)"""

# LR
lr = LogisticRegression(C=100.0, max_iter=7000, random_state=1)
lr.fit(train_data, train_label)
pre = pd.DataFrame(lr.predict(test_data))
accu = acs(test_label, pre)
print('LR ' + 'accuracy: {:.2%}'.format(accu))
"""with open('../model/lr_gesture.pickle', 'wb')as f:
    pk.dump(lr, f)"""

# KNN
knn = KNeighborsClassifier()
knn.fit(train_data, train_label)
pre = pd.DataFrame(knn.predict(test_data))
accu = acs(test_label, pre)
print('KNN ' + 'accuracy: {:.2%}'.format(accu))
"""with open('../model/knn_gesture.pickle', 'wb')as f:
    pk.dump(knc, f)"""

# DTC
dtc = DecisionTreeClassifier()
dtc.fit(train_data, train_label)
pre = pd.DataFrame(dtc.predict(test_data))
accu = acs(test_label, pre)
print('DTC ' + 'accuracy: {:.2%}'.format(accu))
"""with open('../model/dtc_gesture.pickle', 'wb')as f:
    pk.dump(dtc, f)"""

# GNB
gnb = GaussianNB()
gnb.fit(train_data, train_label)
pre = pd.DataFrame(gnb.predict(test_data))
accu = acs(test_label, pre)
print('GNB ' + 'accuracy: {:.2%}'.format(accu))
"""with open('../model/gnb_gesture.pickle', 'wb')as f:
    pk.dump(gnb, f)"""

# QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(train_data, train_label)
pre = pd.DataFrame(qda.predict(test_data))
accu = acs(test_label, pre)
print('QDA ' + 'accuracy: {:.2%}'.format(accu))
"""with open('../model/qda_gesture.pickle', 'wb')as f:
    pk.dump(qda, f)"""

"""
1 frame
MLP accuracy: 65.69%
LR accuracy: 65.99%
KNN accuracy: 46.74%
DTC accuracy: 53.57%
GNB accuracy: 29.23%
QDA accuracy: 53.00%

2 frames
MLP accuracy: 61.20%
LR accuracy: 66.32%
KNN accuracy: 38.42%
DTC accuracy: 51.22%
GNB accuracy: 31.56%
QDA accuracy: 52.84%

3 frames
MLP accuracy: 50.51%
LR accuracy: 67.44%
KNN accuracy: 32.57%
DTC accuracy: 52.94%
GNB accuracy: 26.89%
QDA accuracy: 53.07%

4 frames
MLP accuracy: 49.68%
LR accuracy: 67.91%
KNN accuracy: 30.81%
DTC accuracy: 51.49%
GNB accuracy: 23.09%
QDA accuracy: 53.11%

5 frames
MLP accuracy: 45.00%
LR accuracy: 67.35%
KNN accuracy: 29.26%
DTC accuracy: 52.68%
GNB accuracy: 21.11%
QDA accuracy: 53.16%
"""

