from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def isBorder(x,k,x_train,y_train):
    color=["pink","red","black"]
    d=[np.sqrt(np.sum((x_i-x)**2)) for x_i in x_train]
    nearest=np.argsort(d)
    top_k=[y_train[i] for i in nearest[:k]]
    votes=Counter(top_k)
    if k==1:
        return color[votes.most_common(2)[0][0]]
    if len(votes.most_common(2))>=2:
        if (votes.most_common(2)[0][1]+votes.most_common(2)[1][1])%2==0:
            if votes.most_common(2)[0][1]-votes.most_common(2)[1][1]==0:
                return color[0]
        elif votes.most_common(2)[0][1]-votes.most_common(2)[1][1]<=1:
            return color[votes.most_common(2)[0][0]]
    return False


iris=datasets.load_iris()
x=iris.data
y=iris.target
x=x[y>0]
y=y[y>0]
pca=PCA(n_components=2)
x=pca.fit_transform(x)

#sklearn自带的train_test_split
k=20
x_train,x_test,y_train,y_test=train_test_split(x,y)
knn_classifier=KNeighborsClassifier(k)
knn_classifier.fit(x_train,y_train)
y_predict=knn_classifier.predict(x_test)
scores=knn_classifier.score(x_test,y_test)
print('acc:',scores)
for i in range(-80,80,3):
    for j in range(-100,100,3):
        color=isBorder([j/100,i/100],k,x_train,y_train)
        if (color):
            plt.scatter(j/100,i/100,color=color)
plt.scatter(x[y==1,0],x[y==1,1],color='green')
plt.scatter(x[y==2,0],x[y==2,1],color='blue')
plt.show()