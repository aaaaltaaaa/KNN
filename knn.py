import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
class KNNClassifier:
    def __init__(self,k):
        assert k>=1
        self.k=k
        self._x_train=None
        self._y_train=None

    def fit(self,x_train,y_train):
        self._x_train=x_train
        self._y_train=y_train
        return self

    def _predict(self,x):
        d=[np.sqrt(np.sum((x_i-x)**2)) for x_i in self._x_train]
        nearest=np.argsort(d)
        top_k=[self._y_train[i] for i in nearest[:self.k]]
        votes=Counter(top_k)
        return votes.most_common(1)[0][0]

    def predict(self,X_predict):
        y_predict=[self._predict(x1) for x1 in X_predict]
        return np.array(y_predict)

    def isBorder(self,x):
        color=["pink","red","black"]
        d=[np.sqrt(np.sum((x_i-x)**2)) for x_i in self._x_train]
        nearest=np.argsort(d)
        top_k=[self._y_train[i] for i in nearest[:self.k]]
        votes=Counter(top_k)
        if self.k==1:
            return color[votes.most_common(2)[0][0]]
        if len(votes.most_common(2))>=2:
            if (votes.most_common(2)[0][1]+votes.most_common(2)[1][1])%2==0:
                if votes.most_common(2)[0][1]-votes.most_common(2)[1][1]==0:
                    return color[0]
            elif votes.most_common(2)[0][1]-votes.most_common(2)[1][1]<=1:
                return color[votes.most_common(2)[0][0]]
        return False

    def __repr__(self):
        return 'knn(k=%d):'%self.k

    def score(self,x_test,y_test):
        y_predict=self.predict(x_test)
        return sum(y_predict==y_test)/len(x_test)

def train_test_split(x,y,test_ratio=0.2,seed=None):
    if seed:
        np.random.seed(seed)
    #生成样本随机的序号
    shuffed_indexes=np.random.permutation(len(x))
    #测试集占样本总数的20%
    test_size=int(test_ratio*len(x))
    test_indexes=shuffed_indexes[:test_size]
    train_indexes=shuffed_indexes[test_size:]
    x_test=x[test_indexes]
    y_test=y[test_indexes]
    x_train=x[train_indexes]
    y_train=y[train_indexes]
    return x_train,x_test,y_train,y_test


iris=datasets.load_iris()
x=iris.data
y=iris.target
x=x[y>0]
y=y[y>0]
pca=PCA(n_components=2)
x=pca.fit_transform(x)
plt.scatter(x[y==1,0],x[y==1,1],color='green')
plt.scatter(x[y==2,0],x[y==2,1],color='blue')
X_train,X_test,y_train,y_test=train_test_split(x,y)
my_knn=KNNClassifier(k=10)
my_knn.fit(X_train,y_train)

for i in range(-80,80,3):
    for j in range(-100,100,3):
        color=my_knn.isBorder([j/100,i/100])
        if (color):
            plt.scatter(j/100,i/100,color=color)

y_predict=my_knn.predict(X_test)
score=my_knn.score(X_test,y_test)
print(score)
plt.show()