## keywords
#sklearn #SVM #Python
           
## 任务要求
**Assignment 1: Speech and not-speech detection**

DDL：2017-10-17 Tue.

（1）This assignment is carried out by group. You could choose your teammate freely. Each group consists of at most 3 students.

（2）The ‘training.data’ contains the training data. It is from our project to detect whether a person in a video speaks or not. The features are generated in the following way, which may help you making the most of these features.

1、Get the mouth region M from the origin image based on facial landmark detection.

2、Calculate dense optic flow between mouth region of last frame and the current frame and generate a score S that depicts the motion of mouth.

3、Calculate the parameter V which depicts the degree of mouth opening.

4、For frame i, we also calculate the S and V for its previous and next frames.

5、Hence, we generate a _6_ dimensional feature vector is X=[Si-1 Si Si+1 Vi-1 Vi Vi+1].

6、The label is at the end of each line, where +1 represents speaking, and -1 represents not-speaking.

In the training.data, the ratio of positive examples over negative examples is 1:1. Keep this in mind, for if you find your training error or validation error is larger than 50%, that means your solution learns nothing and performs worse than guessing.

（4）You need to write a program to predict speaking or not speaking.

For convenience to evaluate your grogram, please use this name for your matlab main function:

speakingDetection.m

Note about the interface in your function ‘speakingDetection.m’, it should be:

function predY= speakingDetection (X)

X: The input feature vectors, which is an N*6 matrix, where N is the number of feature vectors.

predY: The output vector to predict labels of X, which is a N*1 vector, and predY(i) = 1 or -1.

Besides MATLAB, you also use Python, as long as you hold the interface protocol above. Note we don’t recommend C/C++.

（5）You can use ANY method to solve this problem.

## 我的代码

```python
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing

def load_data(filename='training.data', filedir='data/'):
    filepath = os.path.join(filedir, filename)
    df = pd.read_csv(filepath, delim_whitespace=True, names=['S_i-1', 'S_i', 'S_i+1', 'V_i-1', 'V_i', 'V_i+1', 'Y_true'], index_col=False)
    # print(df)
    X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()
    return X, y

def plot_2D(X, y):
    fig = plt.figure(num=1, figsize=(10, 10))
    gs=gridspec.GridSpec(6,6) # 设定网格

    for i in range(6):
        for j in range(i, 6):
            ax = fig.add_subplot(gs[i, j])
            ax.scatter(X[:, i], X[:, j], c=y, s=50, cmap='autumn')

    plt.savefig('result/svm-data-2d.png')



def speakingDetection(X):
    pass

# 读入数据
X, y = load_data()
# 归一化
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
# 划分训练、测试数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=0.2)

clf = SVC(C=1, gamma=10, kernel='rbf', decision_function_shape="ovo")
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
print("训练集：", train_score)
test_score = clf.score(X_test, y_test)
print("测试集：",test_score)
print("支持向量个数：", len(clf.support_))

# CScale = [pow(2, i) for i in range(-5, 16)]
# gammas = [pow(2, i) for i in range(-15, 4)]
# for C in CScale:
#     for gamma in gammas:
#         print(f"C={C:.3f}, gamma={gamma:.3f}")
#         clf = SVC(C=C, gamma=gamma, kernel='rbf', decision_function_shape="ovo")
#         clf.fit(X_train, y_train)
#         train_score = clf.score(X_train, y_train)
#         print("训练集：", train_score)
#         test_score = clf.score(X_test, y_test)
#         print("测试集：",test_score)
```

