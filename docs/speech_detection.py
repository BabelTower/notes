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

