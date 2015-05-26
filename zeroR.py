__author__ = 'tianzhang'

from sklearn import metrics

trainfile = open('train.csv')
header = trainfile.next().rstrip().split(',')

y_train = []

for line in trainfile:
    splitted = line.rstrip().split(',')
    label = int(splitted[0])
    y_train.append(label)

trainfile.close()

count0 = 0
count1 = 0

for item in y_train:
    if item == 0:
        count0 += 1
    else:
        count1 += 1

if count0 >= count1:
    choice = 0;
else:
    choice = 1;

p_train = []
for i in range(0, len(y_train)):
    p_train.append(choice)

fpr, tpr, thresholds = metrics.roc_curve(y_train, p_train, pos_label=1)
auc = metrics.auc(fpr,tpr)
print 'AuC score on training data:',auc

# load test data
testfile = open('test.csv')
# ignore the test header
testfile.next()
number = 0
for line in testfile:
    number += 1;
testfile.close()

p_test = []
for i in range(0, number):
    p_test.append(choice)

predfile = open('predict_zeroR.csv','w+')

print >>predfile, "Id,Choice"
for i in range(0, len(p_test)):
    print >>predfile, str(i+1)+','+str(p_test[i])

predfile.close()
