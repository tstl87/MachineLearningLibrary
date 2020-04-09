import numpy as np
import matplotlib.pyplot as plt
from future.utils import iteritems
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt

class KNN(object):
    def __init__(self,k):
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, X):
        y = np.zeros( len(X) )
        distances = pairwise_distances(X, self.X)
        idx = distances.argsort(axis=1)[:, :self.k]
        votes = self.y[idx]
        for i in range(len(X)):
            y[i] = np.bincount(votes[i]).argmax()

        return y
    
    def score(self, X, y):
        P = self.predict(X)
        return np.mean(P == y)
    
    
def visualizeTrainData( X_train, y_train, model):
    # Visualising the Training set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('KNN Classification (Training set)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


def visualizeTestData( X_test, y_test, model ):
    # Visualising the Test set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('KNN Classification (Test set)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()
                
                
                