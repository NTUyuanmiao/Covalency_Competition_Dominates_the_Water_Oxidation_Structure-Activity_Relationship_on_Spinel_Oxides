import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('materials.csv')

x = dataset.drop(['Name', 'Aa-T', 'Ab-T', 'Ba-T', 'Bb-T', 'O-2p', 'M-oct', 'M-tetra', 'Hybri(O-Moct)', 'Hybri(O-Mtet)', 'weak-hybri'], axis=1).values
y = dataset[['O-2p', 'M-oct', 'M-tetra']].values.reshape(-1, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=3)

x_train_all = x_train.copy()
y_train_all = y_train.copy()
x_test_all = x_test.copy()
y_test_all = y_test.copy()

for i in range(7):
    x_train[:, [i, i + 7]] = x_train[:, [i + 7, i]]
    x_test[:, [i, i + 7]] = x_test[:, [i + 7, i]]

x_train_all = np.concatenate((x_train_all, x_train), axis=0)
y_train_all = np.concatenate((y_train_all, y_train), axis=0)
x_test_all = np.concatenate((x_test_all, x_test), axis=0)
y_test_all = np.concatenate((y_test_all, y_test), axis=0)

for i in range(7):
    x_train[:, [i + 14, i + 21]] = x_train[:, [i + 21, i + 14]]
    x_test[:, [i + 14, i + 21]] = x_test[:, [i + 21, i + 14]]

x_train_all = np.concatenate((x_train_all, x_train), axis=0)
y_train_all = np.concatenate((y_train_all, y_train), axis=0)
x_test_all = np.concatenate((x_test_all, x_test), axis=0)
y_test_all = np.concatenate((y_test_all, y_test), axis=0)

for i in range(7):
    x_train[:, [i, i + 7]] = x_train[:, [i + 7, i]]
    x_test[:, [i, i + 7]] = x_test[:, [i + 7, i]]

x_train_all = np.concatenate((x_train_all, x_train), axis=0)
y_train_all = np.concatenate((y_train_all, y_train), axis=0)
x_test_all = np.concatenate((x_test_all, x_test), axis=0)
y_test_all = np.concatenate((y_test_all, y_test), axis=0)

# clf = MLPRegressor(hidden_layer_sizes=(100, 50, 20, 5, 3), max_iter=500, alpha=1e-5, solver='lbfgs', verbose=10, random_state=21, tol=0.000000001,activation='tanh')

clf = RandomForestRegressor()

clf.fit(x_train_all, y_train_all)

y_train_pred = clf.predict(x_train_all)
y_test_pred = clf.predict(x_test_all)

x_train_result = []
x_test_result = []
y_train_result = []
y_test_result = []

for i, j in zip(abs(y_train_all[:,0] - y_train_all[:,1]), abs(y_train_all[:,0] - y_train_all[:,2])):
    x_train_result.append(max(i, j))
for i, j in zip(abs(y_train_pred[:,0] - y_train_pred[:,1]), abs(y_train_pred[:,0] - y_train_pred[:,2])):
    y_train_result.append(max(i, j))
for i, j in zip(abs(y_test_all[:,0] - y_test_all[:,1]), abs(y_test_all[:,0] - y_test_all[:,2])):
    x_test_result.append(max(i, j))
for i, j in zip(abs(y_test_pred[:,0] - y_test_pred[:,1]), abs(y_test_pred[:,0] - y_test_pred[:,2])):
    y_test_result.append(max(i, j))

ytrain = []
ytest = []

l = len(y_train_result) // 4

for i in range(l):
    ytrain.append((y_train_result[i] + y_train_result[i + l] + y_train_result[i + l * 2] + y_train_result[i + l * 3]) / 4)

k = len(y_test_result) // 4

for i in range(k):
    ytest.append((y_test_result[i] + y_test_result[i + k] + y_test_result[i + k * 2] + y_test_result[i + k * 3]) / 4)

print("Train_Data_X:")
print(x_train_result[0:l])
print("-" * 13)
print("Train_Data_Y:")
print(ytrain)
print("-" * 13)
print("Test_Data_X:")
print(x_test_result[0:k])
print("-" * 13)
print("Test_Data_Y:")
print(ytest)

### R2 Score ###
# print(y_pred)
# print(r2_score(x_test_result[0:k],ytest))

### Save the image ###
# df_tr = pd.DataFrame({'DFT calculated':x_train_result[0:l], 'Predicted':ytrain})
# img = sns.scatterplot(x="DFT calculated", y="Predicted", data=df_tr).get_figure()
# df_tr = pd.DataFrame({'DFT calculated':x_test_result[0:k], 'Predicted':ytest})
# img = sns.scatterplot(x="DFT calculated", y="Predicted", data=df_tr).get_figure()
# img.savefig('Test.png')



### Predict the performance of new materials ###
# xrt = [Add_The_Formatted_Data_Here]
# ytp = clf.predict(xrt)
# ytr = []
# yt = []
# for i, j in zip(abs(ytp[:,0] - ytp[:,1]), abs(ytp[:,0] - ytp[:,2])):
#     ytr.append(max(i, j))
# m = len(ytp) // 4
# for i in range(m):
#     yt.append((ytr[i] + ytr[i + m] + ytr[i + m * 2] + ytr[i + m * 3]) / 4)
#
# print(yt)

