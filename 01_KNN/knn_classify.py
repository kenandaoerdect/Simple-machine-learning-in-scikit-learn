import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data = np.array(pd.read_csv('data/data.csv', header=None))
# print(data.shape)
# (1000, 4)

np.random.seed(666)
np.random.shuffle(data)

pre_train = data[:800]
pre_test = data[800:]
# print(pre_train.shape, pre_test.shape)
# (800, 4) (200, 4)

train_data = pre_train[:, :-1]
train_label = pre_train[:, -1].astype(int)
test_data = pre_test[:, :-1]
test_label = pre_test[:, -1].astype(int)
# print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
# (800, 3) (800,) (200, 3) (200,)

# standardization
standardScalar = StandardScaler()
standardScalar.fit(train_data)
train_data_standard = standardScalar.transform(train_data)
test_data_standard = standardScalar.transform(test_data)

knn_classifier = KNeighborsClassifier(n_neighbors=5)  # default n_neighbors = 5
knn_classifier.fit(train_data_standard, train_label)
acc = knn_classifier.score(test_data_standard, test_label)
print(acc)
# 0.96

# knn_classifier.fit(train_data, train_label)
# acc = knn_classifier.score(test_data, test_label)
# print(acc)
# 0.735
