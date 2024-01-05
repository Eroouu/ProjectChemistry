import openpyxl
import math
from math import log
import os
import scipy
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA
import pandas as pd

def HowManyElementsInG(vector, n, e):
    array_ind_g = np.zeros(len(vector))
    array_ind_g[0] = 1
    for i in range(1, len(vector) - n):
        i_is_in_G = True
        for j in range(i):
            if array_ind_g[j] == 1:
                temp_max = max(np.absolute(vector[i: i + n + 1] - vector[j: j + n + 1]))
                if(temp_max < e):
                    i_is_in_G = False
                    break
        if(i_is_in_G is True):
            array_ind_g[i] = 1
    #print(array_ind_g)
    return int(sum(array_ind_g))

def FindSpecFunkValue(vector, n, e):
    C_n_e = HowManyElementsInG(vector, n, e)
    if (C_n_e == 0):
        print("C = 0")
        rez = 0
    else:
        rez = log(C_n_e) / (n - log(e))
    return rez

def FindManyValuesInVector(vector, n, e, proc_decriese, counte):
    array = []
    for i in range(counte):
        array.append(FindSpecFunkValue(vector, n, e))
        e *= (1-proc_decriese )
    return array

def MCM(time_series, k):
    el_count = len(time_series) - k + 1
    data = np.array([time_series[i:i + k] for i in range(el_count)])
    Q = np.cov(data)
    names = [i for i in range(el_count)]
    #sns.heatmap(Q, annot=True, fmt='g', xticklabels=names, yticklabels=names)
    #plt.show()

    eigenvalues, eigenvectors = LA.eig(Q)
    df = pd.DataFrame(eigenvalues)
    print(df)
    x = np.array([i + 1 for i in range(el_count)])
    plt.plot(x, np.log(eigenvalues))
    plt.show()
    return 0
