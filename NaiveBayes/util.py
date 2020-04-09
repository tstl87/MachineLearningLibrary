import numpy as np
import pandas as pd 


def get_xor():

	X = np.zeros((200,2))
	X[:50] = np.random.random((50,2)) / 2 + 0.5 # (0.5-1, 0.5-1)
	X[50:100] = np.random.random((50,2)) / 2    # (0-.5, 0-.5)
	X[100:150] = np.random.random((50,2)) / 2 + np.array([[0,0.5]]) # (0-.5, .5-1)
	X[150:] = np.random.random((50,2)) / 2 + np.array([[0.5,0]]) # (.5-1,0-.5)
	y = np.array([0]*100 + [1]*100)

	return X, y


def get_donut():

	N = 200
	R_inner = 5
	R_outer = 10

	R1 = np.random.randn(N//2)/2 + R_inner
	theta = 2*np.pi*np.random.random( N//2 )
	X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

	R2 = np.random.randn(N//2)/2 + R_outer
	theta = 2*np.pi*np.random.random( N//2 )
	X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

	X = np.concatenate( [ X_inner, X_outer ])
	y = np.array([0]*(N//2) + [1]*(N//2))

	return X, y


def get_swirl():
    N = 200
    theta = 3.5 * np.pi * np.linspace(0,1,N//2)
    R =  np.linspace(0,6,N//2)
    X1 = np.concatenate([[R * np.cos(theta)], [R * np.sin(theta)]]).T
    X2 = np.concatenate([[R * np.cos(theta + np.pi)], [R * np.sin(theta + np.pi)]]).T

    X = np.concatenate( [ X1, X2 ])
    y = np.array([0]*(N//2) + [1]*(N//2))

    return X, y


def get_SBG():

	N = 200

	X1 = np.random.multivariate_normal([4,4], np.eye(2), size= N//2 ) / 2
	X2 = np.random.multivariate_normal([-4,-4], np.eye(2), size = N//2) / 2

	X = np.concatenate( [X1, X2])
	y = np.array([0]*(N//2) + [1]*(N//2))

	return X, y