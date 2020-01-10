import pandas as pd 
import numpy as np
from scipy.linalg import eigh

def scatter(C):
    mu = np.average(C, axis = 0)
    d = len(mu)
    S = np.zeros((d,d))

    for i in range(C.shape[0]):
        x = C[i,:]
        xx = np.outer(x-mu, x-mu)
        S += xx
    return S

# Calculates the S_b and S_w matrices

def scatter_b(X): 
    x_arr = [*X.values()]
    mus = []
    ns = []
    d = x_arr[0].shape[1]

    Sw = np.zeros((d,d))

    for i in range(len(x_arr)):
        xi = x_arr[i]
        ni = xi.shape[0]

        Sw += scatter(xi)

        mu = np.average(xi,axis = 0)
        mus.append(mu)
        ns.append(ni)

    Mu = np.average(mus, axis = 0)

    Sb = np.zeros((len(Mu),len(Mu)))

    for i in range(len(x_arr)):
        Sb += ns[i]*np.outer(mus[i]-Mu,mus[i]-Mu)

    return Sb,Sw

def LDA(df):

    ndata = []
    for c in df.columns:
        if type(df[c][0]) is np.int64:
            ndata.append(c)

    num_data = df[ndata[1:-1]] # Drop index and class label

    data = df[ndata[1:-1]].values # Df in array form
    
    classes = set(df["label"].values) # Unique classes in an array
    data_class = {} # Collect class entries into a dict
    for cl in classes:
        data_class[cl] = (num_data[df.label==cl][ndata[1:-1]]).values

    Sb, Sw = scatter_b(data_class) # Calculate the variance matrices
    

    #Solve generalized eigenvalue equation and sort

    eigvals, eigvecs = eigh(Sb, Sw)

    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[idx]


    k = 0
    s = np.sum(eigvals)
    sk = 0.0
    while sk/s < 0.85: # Find the number of variables responsible for 85% of the variance
        sk += eigvals[k]
        k+=1

    W = eigvecs[:,0:k] # Pick the respective eigenvectors

    X_LDA = np.dot(data,W)

    return X_LDA



if __name__ == '__main__':

    df = pd.read_csv("../data/train.csv")

    LDA(df)
