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
from scipy.linalg import svd
from sklearn.decomposition import PCA

#функция анализирует множество G и выдает сколько по итогу в нем элементов
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
        e *= (1-proc_decriese)
    return array

def Making_Covar_Matrix(time_series, k):
    n = len(time_series)
    series_mean = np.mean(time_series)
    C = np.zeros((k, k))
    for i in range(k):
        ti = time_series[i:i + n - k + 1]
        C[i, i] = np.dot(ti - series_mean, ti - series_mean)
        for j in range(i + 1, k):
            tj = time_series[j:j + n - k + 1]
            tj_mean = np.mean(tj)
            C[i, j] = np.dot(ti - series_mean, tj - series_mean)
            C[j, i] = C[i, j]
    #print(np.trace(C))
    return C
def Making_Covar_Matrix_For_Separeted_Vectors(matrix, k):
    return 0

def Covar_Matrix_Inf(C, k):
    plt.figure(figsize=(9, 7), dpi=80)
    sns.heatmap(C, annot=True, fmt='g', xticklabels=[i for i in range(k)], yticklabels=[i for i in range(k)])
    plt.title(f"Матрица ковариации для k = {k}", fontsize=18)
    plt.show()
    eigenvalues, eigenvectors = LA.eig(C)
    print(f'собственные значения матрицы для k={k}')
    df = pd.DataFrame(eigenvalues)
    print(df)
    x = np.array([i + 1 for i in range(k)])
    #plt.plot(x, np.log(eigenvalues[0:k]))
    plt.plot(x, eigenvalues[0:k])
    plt.title(f"График значений собственных чисел для k = {k}", fontsize=14)
    plt.show()

def The_broken_cane_method(C, k):
    # реализация метода сломаной трости
    eigenvalues, eigenvectors = LA.eig(C)
    trC = np.trace(C)
    t = [1 / i for i in range(1, k + 1)]
    l = np.array([sum(t[i:]) / k for i in range(k)])
    dfdata = {
        'lambda/trC': eigenvalues / trC,
        'li': l,
        'сравнение 1 > 2': np.greater(eigenvalues / trC, l)
    }
    df = pd.DataFrame(dfdata)
    print('\n', df)
    return 0

def PCA(time_series,  k):
    data = np.array([time_series[i:i + len(time_series) - k] for i in range(k)])
    npС = np.cov(data, bias=True)
    c = Making_Covar_Matrix((time_series - np.mean(time_series)) / max(max(time_series), abs(min(time_series))), k)
    #plt.plot(MakingNewVectors(time_series, c, k)[:, 1])
    #plt.show()
    #print(C, '\n', npС)
    Covar_Matrix_Inf(c,k)
    The_broken_cane_method(c, k)
    return 0

def Making_Covar_Matrix_2(time_series, k):
    n = len(time_series)
    series_mean = np.mean(time_series)
    C = np.zeros((n // 50, n // 50))
    for i in range(n // 50):
        ti = time_series[i * 50:50 + i * 50]
        C[i, i] = np.dot(ti - series_mean, ti - series_mean)
        for j in range(i + 1, n // 50):
            tj = time_series[j * 50:50+j * 50]
            C[i, j] = np.dot(ti - series_mean, tj - series_mean)
            C[j, i] = C[i, j]
    #print(np.trace(C))
    return C

def PCA_2(time_series, n):
    C = Making_Covar_Matrix_2(time_series, n)
    #plt.plot(MakingNewVectors_2(time_series, C, n)[:, 1])
    #plt.show()
    #print(C, '\n', npc)
    Covar_Matrix_Inf(C, n)
    The_broken_cane_method(C, n)
    return 0
def GetEigenvectors(time_series, k):
    c = Making_Covar_Matrix(time_series, k)
    eigenvalues, eigenvectors = LA.eig(c)
    return eigenvectors
def MakingNewVectorsInKDimension(time_series, C, k):
    eigenvalues, eigenvectors = LA.eig(C)
    n = len(time_series)
    temp_series = []
    for i in range(n - k):
        t = time_series[i:i + k]
        tt = []
        for j in range(k):
            tt.append(np.dot(t, eigenvectors[j]))
        temp_series.append(tt)
    return np.array(temp_series)
def MakingArrayOfComponentsValue(time_series, k):
    c = Making_Covar_Matrix(time_series, k)
    eigenvalues, eigenvectors = LA.eig(c)
    vectors = np.array([time_series[i:i + len(time_series) - k] for i in range(k)])
    vector_projection = []
    for i in range(k):
        vector_projection.append([np.dot(vectors[j][0:k], eigenvectors[i]) for j in range(k)])
    return vector_projection

def MakingArrayOfComponentsValue_2(time_series, k):
    vectors = np.array([time_series[i * 50:i * 50 + 50] - np.mean(time_series) for i in range(len(time_series) // 50)])
    print(len(vectors[0, 0:k]))
    c = np.cov(vectors)
    eigenvalues, eigenvectors = LA.eig(c)
    vector_projection = []
    for i in range(len(time_series) // 100):
        #vector_projection.append([np.dot(vectors[j][0:k], eigenvectors[0:k][j]) for j in range(len(time_series) // 100)])
        #print(len(vectors[i, 0:k]), vectors[i, 0:k])
        #print(len(eigenvectors[0:k, i]), eigenvectors[0:k, i])
        print('ku')
    return vector_projection