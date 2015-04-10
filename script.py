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
    #mat = np.sort(mat, axis=0)

    # Split the matrix up by the labels.
    a = mat[mat[:, 2] == 1.0, :]
    b = mat[mat[:, 2] == 2.0, :]
    c = mat[mat[:, 2] == 3.0, :]
    d = mat[mat[:, 2] == 4.0, :]
    e = mat[mat[:, 2] == 5.0, :]

    # Strip off the last column
    a = a[:, :2]
    b = b[:, :2]
    c = c[:, :2]
    d = d[:, :2]
    e = e[:, :2]

    # Build the list of size k of dxd matrices.
    covmats = []
    covmats.append(np.cov(a.T))
    covmats.append(np.cov(b.T))
    covmats.append(np.cov(c.T))
    covmats.append(np.cov(d.T))
    covmats.append(np.cov(e.T))


    return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value

    # IMPLEMENT THIS METHOD


    a = 1/np.sqrt(2 * np.pi) * (np.linalg.det(covmat)**2)
    pdfs = np.ones((Xtest.shape[0],means.shape[1]))

    for i in range(0, Xtest.shape[0]):

        for q in range(0, means.shape[1]):

            b = (np.transpose(Xtest[i, :] - means[:, q].T))
            b = np.asmatrix(b)
            c = np.linalg.inv(covmat)
            d = b.T
            upper = np.dot(b, c)
            upper = -.5 * np.dot(upper, d)
            pdf_val = np.exp(upper[0,0]) * a
            pdfs[i, q] = pdf_val

    maxs = np.argmax(pdfs, axis=1) + 1

    maxs = np.asmatrix(maxs).T
    acc = 0
    for x in range(0,Xtest.shape[0]):
        if maxs[x, 0] == ytest[x, 0]:
            acc= acc+1
    total = Xtest.shape[0] * 1.0
    acc = acc * 1.0


    return acc/total, maxs


def qdaTest(means, covmats, Xtest, ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value

    # IMPLEMENT THIS METHOD

    pdfs = np.ones((Xtest.shape[0],means.shape[1]))

    for i in range(0, Xtest.shape[0]):

        for q in range(0, means.shape[1]):

            b = (np.transpose(Xtest[i, :] - means[:, q].T))
            b = np.asmatrix(b)
            c = np.linalg.inv(covmats[q])
            d = b.T
            upper = np.dot(b, c)
            upper = -.5 * np.dot(upper, d)
            a = 1/np.sqrt(2 * np.pi) * (np.linalg.det(covmats[q])**2)
            pdf_val = np.exp(upper[0,0]) * a
            pdfs[i, q] = pdf_val

    maxs = np.argmax(pdfs, axis=1) + 1

    maxs = np.asmatrix(maxs).T
    acc = 0
    for x in range(0,Xtest.shape[0]):
        if maxs[x, 0] == ytest[x, 0]:
            acc= acc+1
    total = Xtest.shape[0] * 1.0
    acc = acc * 1.0

    return acc/total, maxs



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
    I = np.identity(a.shape[0]) * (lambd*X.shape[0])
    c = np.linalg.inv(I + a)

    w = np.dot(c, b)

    return w


def testOLERegression(w, Xtest, ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # rmse

    # IMPLEMENT THIS METHOD
    # see pdf ... missing sqrt
    #for i in range(0,Xtest.shape[0]):
    #    rmse = rmse + (np.subtract(ytest[i], np.dot(w.T, Xtest[i]))**2)
    #rmse = sum((ytest[x] - np.dot(w.T, Xtest[x]))**2 for x in range(0, Xtest.shape[0]))
    # print np.asmatrix(np.sqrt(rmse) / Xtest.shape[0])
    #np.asmatrix(np.sqrt(rmse) / Xtest.shape[0]).shape
    #return np.asmatrix(np.sqrt(rmse) / Xtest.shape[0])
    a = (np.sqrt(np.dot(np.transpose(ytest - np.dot(Xtest,w)),(ytest - np.dot(Xtest,w)))))/Xtest.shape[0]
    #a = (np.sqrt(np.square(ytest - np.dot(w.T, Xtest))))/Xtest.shape[0]
    return a





def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD
    w = np.asmatrix(w).T

    N = X.shape[0]
    d = lambd * np.dot(w.T, w)
    b = np.square((y - np.dot(X, w)))
    #a = b.T
    error = np.sum(b) / N + d


    a = np.dot(X,w) - y
    b = np.dot(X.T,a)
    c = 2*lambd*w

    error_grad = (2*b)/N + c

    error_grad = np.squeeze(np.asarray(error_grad))



    return error[0,0], error_grad


def mapNonLinear(x, p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    # IMPLEMENT THIS METHOD

    Xd = np.ones((x.shape[0], p+1))
    for j in range (0,x.shape[0]):
        for i in range(0, p+1):
            Xd[j,i] = x[j] ** i
    return Xd

# Main script

# Problem 1
# load the sample data
X,y,Xtest,ytest = pickle.load(open('sample.pickle', 'rb'))
# LDA
means,covmat = ldaLearn(X,y)
ldaacc, maxs = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc, maxs2 = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# x1 = np.linspace(min(X[:,0]),max(X[:,0]),100)
# y1 = np.linspace(min(X[:,1]), max(X[:,1]),100)
# X1, Y1 = np.meshgrid(x1,y1)
# n = x1.shape[0]*y1.shape[0]
# D = np.zeros((n,2))
# D[:,0] = X1.ravel();
# D[:,1] = Y1.ravel();
#
# comb = np.hstack((Xtest,maxs))
# comb = np.asarray(comb)
#
# ones = comb[comb[:, 2] == 1.0, :]
# twos = comb[comb[:,2] == 2.0,:]
# threes = comb[comb[:,2] == 3.0,:]
# fours = comb[comb[:,2] == 4.0,:]
# fives = comb[comb[:,2] == 5.0,:]
#
# L = np.zeros((n,1));
#
#
# for i in range(n):
#     if D[i,0] >= min(ones[:,0]) and D[i,0] <=max(ones[:,0]) and D[i,1] >=min(ones[:,1]) and D[i,1] <=max(ones[:,1]):
#         L[i] = 1
#     elif D[i,0] >= min(twos[:,0]) and D[i,0] <=max(twos[:,0]) and D[i,1] >=min(twos[:,1]) and D[i,1] <=max(twos[:,1]):
#         L[i] = 2
#     elif D[i,0] >= min(threes[:,0]) and D[i,0] <=max(threes[:,0]) and D[i,1] >=min(threes[:,1]) and D[i,1] <=max(threes[:,1]):
#         L[i] = 3
#     elif D[i,0] >= min(fours[:,0]) and D[i,0] <=max(fours[:,0]) and D[i,1] >=min(fours[:,1]) and D[i,1] <=max(fours[:,1]):
#         L[i] = 4
#     elif D[i,0] >= min(fives[:,0]) and D[i,0] <=max(fives[:,0]) and D[i,1] >=min(fives[:,1]) and D[i,1] <=max(fives[:,1]):
#         L[i] = 5
#
# labels = L.reshape(x1.shape[0], y1.shape[0]);
# plt.contourf(x1,y1,labels)
# plt.show()
#
# comb = np.hstack((Xtest,maxs2))
# comb = np.asarray(comb)
#
# ones = comb[comb[:, 2] == 1.0, :]
# twos = comb[comb[:,2] == 2.0,:]
# threes = comb[comb[:,2] == 3.0,:]
# fours = comb[comb[:,2] == 4.0,:]
# fives = comb[comb[:,2] == 5.0,:]
#
# L = np.zeros((n,1));
#
# for i in range(n):
#     if D[i,0] >= min(ones[:,0]) and D[i,0] <=max(ones[:,0]) and D[i,1] >=min(ones[:,1]) and D[i,1] <=max(ones[:,1]):
#         L[i] = 1
#     elif D[i,0] >= min(twos[:,0]) and D[i,0] <=max(twos[:,0]) and D[i,1] >=min(twos[:,1]) and D[i,1] <=max(twos[:,1]):
#         L[i] = 2
#     elif D[i,0] >= min(threes[:,0]) and D[i,0] <=max(threes[:,0]) and D[i,1] >=min(threes[:,1]) and D[i,1] <=max(threes[:,1]):
#         L[i] = 3
#     elif D[i,0] >= min(fours[:,0]) and D[i,0] <=max(fours[:,0]) and D[i,1] >=min(fours[:,1]) and D[i,1] <=max(fours[:,1]):
#         L[i] = 4
#     elif D[i,0] >= min(fives[:,0]) and D[i,0] <=max(fives[:,0]) and D[i,1] >=min(fives[:,1]) and D[i,1] <=max(fives[:,1]):
#         L[i] = 5
#
# labels = L.reshape(x1.shape[0], y1.shape[0]);
# plt.contourf(x1,y1,labels)
# plt.show()

# Problem 2

X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
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
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
rmses3_1 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    #w_l_1 = learnRidgeRegression(X,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    #rmses3_1[i] = testOLERegression(w_l,X_i,y)
    i = i + 1
# plt.plot(lambdas,rmses3)
# plt.show()
# #Plot for data
# plt.plot(lambdas,rmses3_1)
# plt.show()

#print "End Of Problem 3"
# Problem 4
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
#rmses4_1 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    #rmses4_1[i] = testOLERegression(w_l_1, X_i,y)
    i = i + 1
# plt.plot(lambdas,rmses4)
# plt.show()
# plt.plot(lambdas,rmses4_1)
# plt.show()

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
#rmses5_1 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    #rmses5_1[p,0] = testOLERegression(w_d1,Xd,y)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    #rmses5_1[p,1] = testOLERegression(w_d2,Xd,y)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
#
# plt.plot(range(pmax),rmses5)
# plt.legend(('No Regularization','Regularization'))
# plt.show()
#
# plt.plot(range(pmax), rmses5_1)
# plt.legend(('No Regularization', 'Regularization'))
# plt.show()'

print "Finished"