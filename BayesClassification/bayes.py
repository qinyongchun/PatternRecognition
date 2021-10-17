import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)

#-------------------------------------------------------------------------
#   load data from csv
data_train = np.array(pd.read_csv(r'data/train_data.csv',header=None))
data_test = np.array(pd.read_csv(r'data/test_data.csv',header=None))

#-------------------------------------------------------------------------
#   data segmentation based on labels
for i in range(0,data_train.shape[0]):
    if data_train[i][0]==2:
        label1 = i-1
        break
for j in range(i,data_train.shape[0]):
    if data_train[j][0]==3:
        label2 = j-1
        break

sample1=data_train[0:label1,1:data_train.shape[1]]
sample2=data_train[label1+1:label2,1:data_train.shape[1]]
sample3=data_train[label2+1:data_train.shape[0],1:data_train.shape[1]]

#-------------------------------------------------------------------------
#   calculate mean value and standard deviation of each feature
mean1=sample1.mean(axis=0)
sigma1=sample1.std(axis=0,ddof=1)
# np.savetxt("mean1.csv", mean1, delimiter=',')
# np.savetxt("sigma1.csv", sigma1, delimiter=',')

mean2=sample2.mean(axis=0)
sigma2=sample2.std(axis=0,ddof=1)
# np.savetxt("mean2.csv", mean2, delimiter=',')
# np.savetxt("sigma2.csv", sigma2, delimiter=',')

mean3=sample3.mean(axis=0)
sigma3=sample3.std(axis=0,ddof=1)
# np.savetxt("mean3.csv", mean3, delimiter=',')
# np.savetxt("sigma3.csv", sigma3, delimiter=',')

#-------------------------------------------------------------------------
#   calculate posterior probability based on 
#   class conditional probability density and prior probability

#   class 1
p=np.ones((data_test.shape[0],3))
for i in range(0,data_test.shape[0]):
    x = data_test[i,1:data_test.shape[1]]
    for j in range(0,data_test.shape[1]-1):
        m=normal_distribution(x[j],mean1[j],sigma1[j])
        p[i,0]=p[i,0]*normal_distribution(x[j],mean1[j],sigma1[j])
    p[i,0]=p[i,0]*(label1+1)/(data_train.shape[0]+1)

#   class 2
for i in range(0,data_test.shape[0]):
    x = data_test[i,1:data_test.shape[1]]
    for j in range(0,data_test.shape[1]-1):
        p[i,1]=p[i,1]*normal_distribution(x[j],mean2[j],sigma2[j])
    p[i,1]=p[i,1]*(label2-label1)/(data_train.shape[0]+1)

#   class 3
for i in range(0,data_test.shape[0]):
    x = data_test[i,1:data_test.shape[1]]
    for j in range(0,data_test.shape[1]-1):
        p[i,2]=p[i,2]*normal_distribution(x[j],mean3[j],sigma3[j])
    p[i,2]=p[i,2]*(data_train.shape[0]-label2)/(data_train.shape[0]+1)

#-------------------------------------------------------------------------
#   predict and normalize
result=np.zeros((data_test.shape[0],4))
count_correct=0
for i in range(0,data_test.shape[0]):
    #   search the maximum probability
    result[i][0]=format(np.argmax(p[i])+1,'.0f')
    #   normalization
    sum_line=np.sum(p[i])
    result[i,1]=format(p[i,0]/sum_line,'.2f')
    result[i,2]=format(p[i,1]/sum_line,'.2f')
    result[i,3]=format(p[i,2]/sum_line,'.2f')
    #   count accuracy
    if result[i][0]==data_test[i][0]:
        count_correct=count_correct+1

print("test_prediction:")
print(result)
print("Classification accuracy:",count_correct/data_test.shape[0])
# np.savetxt("test_prediction.csv", result, delimiter=',')
