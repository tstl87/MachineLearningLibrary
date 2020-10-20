# Standard linear model class
import numpy as np
import pandas as pd
import random


class LinearModel():
  def __init__(self, degree = 1, regularization = None, Lambda = 0, stepSize = 0.01):
    self.m = [] # coefficients of linear model
    self.d = degree # degree of linear model if using a polynomial
    self.reg = regularization # type of regularization term. Input should be either 'l1', 'l2', or None
    self.lam = Lambda
    self.stepSize = stepSize

  def fit(self, train_data, y_train):

    X_train = np.ones( (len(train_data),self.d+1) )

    for d in range( 1, self.d+1):
      X_train[:,d] = [ val**d for val in train_data ]

    self.gradientDescent(X_train,y_train)

    print( self.ssr(X_train, y_train) )

    return
    

  def predict(self, test_data):
    
    X_test = np.ones( (len(test_data),self.d+1) )

    for d in range(1,seld.d+1):
      X_test[:,d] = data**d

    return np.dot(X_test, self.m)
    

  def gradientDescent(self, x, y):

    m_old = np.random.normal(0, 1/len(x[0]), size = (self.d+1) )
    m_new = 10000*np.random.normal(0, 1/len(x[0]), size = (self.d+1) )
        
    iterations = 0
    print('m_old = ', m_old)
    print('m_new = ', m_new)

    #np.linalg.norm(m_new - m_old)/np.linalg.norm(m_new) > 1e-9

    while( iterations < 999 ):

      iterations += 1
      m_new = m_old - self.stepSize*self.dssr(x,y,m_old)
      m_old = m_new.copy()

    self.m = m_new

    return
    

  def ssr(self, x,y):

    rv = np.dot( np.dot(x,self.m) - y, np.dot(x,self.m) - y )
    if not self.reg:
      # sum( (x*m - y)^2 )
      return rv
    elif self.reg == 'l1':
      # sum( (x*m - y)^2 ) + lam*sum(m^2)
      return rv + sum([self.lam*abs(val) for val in self.m])
    elif self.reg == 'l2':
      # sum( (x*m - y)^2) + lam*sum(abs(m))
      return rv + self.lam*np.dot(  self.m, self.m  )
    else:
      print('error: invalid input for regularization')

  def dssr(self, x,y,m):

    if not self.reg:
      #  2x^T*(x*m-y)
      return  np.dot(2*np.transpose(x), np.dot(x,m)-y)
    elif self.reg == 'l1':
      #  2x^T*(x*m-y) + lambda*sign(m)
      return np.dot(2*np.transpose(x), np.dot(x,m)-y) + self.lam*np.sign(m)
    elif self.reg == 'l2':
      # 2x^T*(x*m-y) + 2*lambda*m
      return np.dot(2*np.transpose(x), np.dot(x,m)-y) + self.lam*2*m
    else:
      print('error: invalid input for regularization')

# sample data
data = {'x':[-2,-1,0,1,2], 'y':[5,2,1,2,5]}
data = pd.DataFrame(data=data)
test = {'x':[-4,-3,3,4], 'y':[17,10,10,17]}
test = pd.DataFrame(data=test)

lm = LinearModel(degree=2, regularization='l1', Lambda=1)
lm.fit(data['x'], data['y'])

print(lm.m)
