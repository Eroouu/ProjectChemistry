from math import log
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA
import pandas as pd
from scipy.linalg import svd
from sklearn.decomposition import PCA


def how_many_elements_in_g(vector, n, e):
    """
    функция анализирует множество G и выдает сколько по итогу в нем элементов
    :param vector:
    :param n:
    :param e:
    :return: количество элементов входящих в G
    """
    array_ind_g = np.zeros(len(vector))
    array_ind_g[0] = 1
    for i in range(1, len(vector) - n):
        i_is_in_g = True
        for j in range(i):
            if array_ind_g[j] == 1:
                temp_max = max(np.absolute(vector[i: i + n + 1] - vector[j: j + n + 1]))
                if temp_max < e:
                    i_is_in_g = False
                    break
        if i_is_in_g is True:
            array_ind_g[i] = 1
    return int(sum(array_ind_g))


def find_spec_funk_value(vector, n, e):
    c_n_e = how_many_elements_in_g(vector, n, e)
    if c_n_e == 0:
        print("C = 0")
        rez = 0
    else:
        rez = log(c_n_e) / (n - log(e))
    return rez


def find_many_values_in_vector(vector, n, e, proc_decrease, count_e):
    array = []
    for i in range(count_e):
        array.append(find_spec_funk_value(vector, n, e))
        e *= (1-proc_decrease)
    return array


def making_covar_matrix(time_series, k):
    n = len(time_series)
    series_mean = np.mean(time_series)
    covmat = np.zeros((k, k))
    for i in range(k):
        ti = time_series[i:i + n - k + 1]
        covmat[i, i] = np.dot(ti - series_mean, ti - series_mean)
        for j in range(i + 1, k):
            tj = time_series[j:j + n - k + 1]
            covmat[i, j] = np.dot(ti - series_mean, tj - series_mean)
            covmat[j, i] = covmat[i, j]
    return covmat


def covar_matrix_information(covmat, k):
    plt.figure(figsize=(9, 7), dpi=80)
    sns.heatmap(covmat, annot=True, fmt='g', xticklabels=[i for i in range(k)], yticklabels=[i for i in range(k)])
    plt.title(f"Матрица ковариации для k = {k}", fontsize=18)
    plt.show()
    eigenvalues, eigenvectors = LA.eig(covmat)
    print(f'собственные значения матрицы для k={k}')
    df = pd.DataFrame(eigenvalues)
    print(df)
    x = np.array([i + 1 for i in range(k)])
    plt.plot(x, eigenvalues[0:k])
    plt.title(f"График значений собственных чисел для k = {k}", fontsize=14)
    plt.show()


def the_broken_cane_method(covmat, k):
    # реализация метода сломаной трости
    eigenvalues, eigenvectors = LA.eig(covmat)
    trace_c = np.trace(covmat)
    t = [1 / i for i in range(1, k + 1)]
    length = np.array([sum(t[i:]) / k for i in range(k)])
    dfdata = {
        'lambda/trC': eigenvalues / trace_c,
        'li': length,
        'сравнение 1 > 2': np.greater(eigenvalues / trace_c, length)
    }
    df = pd.DataFrame(dfdata)
    print('\n', df)
    return 0


def PCA(time_series,  k):
    data = np.array([time_series[i:i + len(time_series) - k] for i in range(k)])
    npcovmatrix = np.cov(data, bias=True)
    c = making_covar_matrix((time_series - np.mean(time_series)) / max(max(time_series), abs(min(time_series))), k)
    '''
    #plt.plot(MakingNewVectors(time_series, c, k)[:, 1])
    #plt.show()
    #print(C, '\n', npС)
    '''
    covar_matrix_information(c, k)
    the_broken_cane_method(c, k)
    return 0


def making_covar_matrix_2(time_series):
    n = len(time_series)
    series_mean = np.mean(time_series)
    covmat = np.zeros((n // 50, n // 50))
    for i in range(n // 50):
        ti = time_series[i * 50:50 + i * 50]
        covmat[i, i] = np.dot(ti - series_mean, ti - series_mean)
        for j in range(i + 1, n // 50):
            tj = time_series[j * 50:50+j * 50]
            covmat[i, j] = np.dot(ti - series_mean, tj - series_mean)
            covmat[j, i] = covmat[i, j]
    return covmat


def PCA_2(time_series, n):
    covmat = making_covar_matrix_2(time_series, n)
    covar_matrix_information(covmat, n)
    the_broken_cane_method(covmat, n)
    return 0


def get_eigenvectors(time_series, k):
    c = making_covar_matrix(time_series, k)
    eigenvalues, eigenvectors = LA.eig(c)
    return eigenvectors


def make_new_vectors_in_k_dimension(time_series, covmatrix, k):
    eigenvalues, eigenvectors = LA.eig(covmatrix)
    n = len(time_series)
    temp_series = []
    for i in range(n - k):
        t = time_series[i:i + k]
        tt = []
        for j in range(k):
            tt.append(np.dot(t, eigenvectors[j]))
        temp_series.append(tt)
    return np.array(temp_series)


def make_array_of_components_value(time_series, k):
    c = making_covar_matrix(time_series, k)
    eigenvalues, eigenvectors = LA.eig(c)
    vectors = np.array([time_series[i:i + len(time_series) - k] for i in range(k)])
    vector_projection = []
    for i in range(k):
        vector_projection.append([np.dot(vectors[j][0:k], eigenvectors[i]) for j in range(k)])
    return vector_projection


def make_array_of_components_value_2(time_series, k):
    vectors = np.array([time_series[i * 50:i * 50 + 50] - np.mean(time_series) for i in range(len(time_series) // 50)])
    print(len(vectors[0, 0:k]))
    c = np.cov(vectors)
    eigenvalues, eigenvectors = LA.eig(c)
    vector_projection = []
    for i in range(len(time_series) // 100):
        vector_projection.append([np.dot(vectors[j][0:k], eigenvectors[0:k][j]) for j in range(len(time_series) // 100)])
        print(len(vectors[i, 0:k]), vectors[i, 0:k])
        print(len(eigenvectors[0:k, i]), eigenvectors[0:k, i])
    return vector_projection