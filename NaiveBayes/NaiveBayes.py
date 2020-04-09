import numpy as np
from future.utils import iteritems
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt


class NaiveBayes(object):
    
    def fit(self, X, y, smoothing = 1e-2):
        self.gaussians = {}
        self.priors = {}
        labels = set(y)
        for c in labels:
            current_X = X[y==c]
            self.gaussians[c] = {
                    'mean': current_X.mean(axis=0),
                    'var': current_X.var(axis=0) + smoothing,
                    }
            self.priors[c] = float( len(y[y==c])) / len(y)
    
    def predict(self, X):
        
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N,K))
        for c,g in iteritems(self.gaussians):
            mean, var = g['mean'], g['var']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
        return np.argmax(P,axis=1)
        
    def score(self, X, y):
        P = self.predict(X)
        return np.mean(P==y)

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
    plt.title('Naive Bayes Classification (Training set)')
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
    plt.title('Naive Bayes Classification (Test set)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()