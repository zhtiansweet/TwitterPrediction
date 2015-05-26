__author__ = 'tianzhang'

import numpy as np
from sklearn import metrics

trainfile = open('train.csv')
header = trainfile.next().rstrip().split(',')

y_train = []
X_train_A = []
X_train_B = []

for line in trainfile:
    splitted = line.rstrip().split(',')
    label = int(splitted[0])
    A_features = [float(item) for item in splitted[1:12]]
    B_features = [float(item) for item in splitted[12:]]

    y_train.append(label)
    X_train_A.append(A_features)
    X_train_B.append(B_features)
trainfile.close()

X_train = []
for i in range(0, len(X_train_A)):
    features = []
    for j in range(0, len(X_train_A[0])):
        sum = float(X_train_A[i][j])+float(X_train_B[i][j])
        dif = float(X_train_A[i][j])-float(X_train_B[i][j])
        if sum == 0:
            feature = 0.0
        else:
            feature = dif/sum
        features.append(feature)
    total = 0
    for item in features:
        total += item
    X_train.append(total)


p_train = []
for i in range(0, len(X_train)):
    if X_train[i] > 0.0:
        choice = 1
    else:
        choice = 0
    p_train.append(choice)

y_train = np.array(y_train)
p_train = np.array(p_train)
fpr, tpr, thresholds = metrics.roc_curve(y_train, p_train, pos_label=1)
auc = metrics.auc(fpr,tpr)
print 'AuC score on training data:',auc



testfile = open('test.csv')
header = testfile.next().rstrip().split(',')

X_test_A = []
X_test_B = []

for line in testfile:
    splitted = line.rstrip().split(',')
    A_features = [float(item) for item in splitted[0:11]]
    B_features = [float(item) for item in splitted[11:]]
    X_test_A.append(A_features)
    X_test_B.append(B_features)
testfile.close()

X_test = []
for i in range(0, len(X_test_A)):
    features = []
    for j in range(0, len(X_test_A[0])):
        sum = float(X_test_A[i][j])+float(X_test_B[i][j])
        dif = float(X_test_A[i][j])-float(X_test_B[i][j])
        if sum == 0:
            feature = 0.0
        else:
            feature = dif/sum
        features.append(feature)
    total = 0
    for item in features:
        total += item
    X_test.append(total)

p_test = []
for i in range(0, len(X_test)):
    if X_test[i] > 0.0:
        choice = 1
    else:
        choice = 0
    p_test.append(choice)

predfile = open('predict_naiveGuess.csv','w+')

print >>predfile, "Id,Choice"
for i in range(0, len(p_test)):
    print >>predfile, str(i+1)+','+str(p_test[i])

predfile.close()
