import numpy as np
from scipy.optimize import minimize
import scipy.ndimage as ndimag
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle


def ldaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD

    ########
    # MEAN #
    ########

    left, right = np.hsplit(X, 2)
    index = np.unique(y)
    xAverage = ndimag.mean(left, labels=y, index=index)
    yAverage = ndimag.mean(right, labels=y, index=index)

    means = np.vstack((xAverage, yAverage))

    ##########
    # COVMAT #
    ##########

    covmat = np.cov(X.T)

    return means, covmat


def qdaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD

    ########
    # MEAN #
    ########
    left, right = np.hsplit(X, 2)
    index = np.unique(y)
    xAverage = ndimag.mean(left, labels=y, index=index)
    yAverage = ndimag.mean(right, labels=y, index=index)
    means = np.vstack((xAverage, yAverage))

    ##########
    # COVMAT #
    ##########

    # Attach the labels to the matrix and sort by labels.
    mat = np.hstack((X, y))
    mat = np.sort(mat, axis=0)

    # Split the matrix up by the labels.
    a = mat[mat[:, 2] == 1, :]
    b = mat[mat[:, 2] == 2, :]
    c = mat[mat[:, 2] == 3, :]
    d = mat[mat[:, 2] == 4, :]
    e = mat[mat[:, 2] == 5, :]

    # Strip off the last column
    a = a[:, :2]
    b = b[:, :2]
    c = c[:, :2]
    d = d[:, :2]
    e = e[:, :2]

    # Build the list of size k or dxd matrices.
    covmats = []
    covmats.append(np.cov(a.T))
    covmats.append(np.cov(b.T))
    covmats.append(np.cov(c.T))
    covmats.append(np.cov(d.T))
    covmats.append(np.cov(e.T))
    print covmats

    return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value

    # IMPLEMENT THIS METHOD
    return 1


def qdaTest(means, covmats, Xtest, ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value

    # IMPLEMENT THIS METHOD
    return 1


def learnOLERegression(X, y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    # IMPLEMENT THIS METHOD

    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))


def learnRidgeRegression(X, y, lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    a = np.dot(X.T, X)
    b = np.dot(X.T, y)
    I = np.identity(a.shape[0]) * lambd
    c = np.linalg.inv(I - a)

    w = np.dot(c, b)

    return w


def testOLERegression(w, Xtest, ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse

    # IMPLEMENT THIS METHOD
    # see pdf ... missing sqrt
    return np.sqrt(np.dot(np.transpose(np.sub(ytest, np.dot(Xtest, w))), np.sub(ytest, np.dot(Xtest, w))))/ Xtest[0]


def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD

    error = np.add(np.div(np.dot(np.transpose(np.sub(y, np.dot(X, w))), np.sub(y, np.dot(X, w))), np.mul(2, X[0])),
                   np.mul(np.mul(np.dot(w.transpose, w), lambd), .5))

    a = np.dot(X.T, X)
    b = np.dot(X.T, y)
    I = np.identity(a.shape[0]) * lambd
    c = np.linalg.inv(I - a)

    error_grad = np.dot(c, b)

    return error, error_grad


def mapNonLinear(x, p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    # IMPLEMENT THIS METHOD

    Xd = np.ones(x.shape[0], p+1)

    for i in range(0, p):
        Xd[:, i]**[i]
    print Xd[1, :]
    return Xd

# Main script

# Problem 1
# load the sample data
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# Problem 2

X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)

# Problem 4
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lamda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))