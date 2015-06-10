__author__ = 'tianzhang'


__author__ = 'tianzhang'

from sklearn import metrics
from sklearn import ensemble
import numpy as np


# load training data
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

y_train = np.array(y_train)
X_train_A = np.array(X_train_A)
X_train_B = np.array(X_train_B)

def transform_features(x):
    return np.log(1+x)

#X_train = transform_features(X_train_A) - transform_features(X_train_B)
#X_train = np.append(X_train, transform_features(X_train_A), axis=1)
#X_train = np.append(X_train, transform_features(X_train_B), axis=1)
X_train = np.append(transform_features(X_train_A), transform_features(X_train_B), axis=1)
#print X_train

model = ensemble.GradientBoostingClassifier(n_estimators=110, max_depth=3)
model.fit(X_train,y_train)

# compute AuC score on the training data
p_train = model.predict_proba(X_train)
p_train = p_train[:,1:2]
#print p_train
fpr, tpr, thresholds = metrics.roc_curve(y_train, p_train, pos_label=1)
auc = metrics.auc(fpr,tpr)
print 'AuC score on training data:',auc

# load test data
testfile = open('test.csv')
#ignore the test header
testfile.next()
X_test_A = []
X_test_B = []
for line in testfile:
    splitted = line.rstrip().split(',')
    A_features = [float(item) for item in splitted[0:11]]
    B_features = [float(item) for item in splitted[11:]]
    X_test_A.append(A_features)
    X_test_B.append(B_features)
testfile.close()

X_test_A = np.array(X_test_A)
X_test_B = np.array(X_test_B)

# transform features in the same way as for training to ensure consistency
#X_test = transform_features(X_test_A) - transform_features(X_test_B)
#X_test = np.append(X_test, transform_features(X_test_A), axis=1)
#X_test = np.append(X_test, transform_features(X_test_B), axis=1)
X_test = np.append(transform_features(X_test_A), transform_features(X_test_B), axis=1)
#print X_test
# compute probabilistic predictions
p_test = model.predict_proba(X_test)
p_test = p_test[:,1:2]

# write predictions
predfile = open('predict_gradientBoosting.csv','w+')

print >>predfile, "Id,Choice"
i=1
for line in p_test:
    print >>predfile, str(i)+','+str(line[0])
    i+=1

predfile.close()

