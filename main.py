import openpyxl
import math
import os
import scipy
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import TackensMethof as tm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def FindFileFrom(name,x ,y):
    wb = openpyxl.load_workbook(filename=name)
    sheet = wb['1']
    time_series = []
    time_array = []
    while sheet.cell(row=x, column=y).value != None:
        time_series.append(float(sheet.cell(row=x, column=y).value))
        time_array.append(float(sheet.cell(row=x, column=y - 1).value))
        x += 1
    return np.array(time_series)

def CheckingTheProbabilityToAnalis(matrix, n, numn, nume, epsilon, ind_start, ind_end):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(projection= "3d")
    nt = np.array([n + i * 5 for i in range(numn)])
    et = np.array([epsilon * 0.85 ** j for j in range(nume)])
    rez_arr = np.array([[tm.FindSpecFunkValue(matrix[ind_start: ind_end], nt[i], et[j]) for j in range(nume)] for i in range(int(numn))])

    ax.scatter(np.repeat(nt, nume), np.tile(et, numn), rez_arr.ravel())
    ax.set_xlabel("n")
    ax.set_ylabel("e")
    plt.title('grapthics for Cne')
    plt.show()

def ex4(matrix, n, numn, nume, epsilon, ind_start, ind_end):
    fig = plt.figure(figsize=(12, 7))
    nt = np.array([n + i * 15 for i in range(numn)])
    et = np.array([epsilon * 0.85 ** j for j in range(nume)])
    rez_arr = np.array([[tm.FindSpecFunkValue(matrix[ind_start: ind_end], nt[i], et[j]) for j in range(nume)] for i in range(int(numn))])
    plt.plot(et, rez_arr[0], 'r', label='n = 20')
    plt.plot(et, rez_arr[1], 'g', label='n = 35')
    plt.plot(et, rez_arr[2], 'b', label='n = 50')
    plt.plot(et, rez_arr[3], 'y', label='n = 65')
    plt.plot(et, rez_arr[4], 'o', label='n = 80')
    plt.legend(bbox_to_anchor=(1.05, 2), loc=2, borderaxespad=0.)
    plt.show()

def TryToFindMC(full_series, k, ind_start, ind_end):
    matrix = np.array(full_series[ind_start:ind_end])
    tm.PCA(matrix, k)
def RandomSeries(a, b, length):
    return np.random.rand(length) * (b - a) + a

def Logistic_Model(x0, numx):
    mu = 4
    massive = []
    massive.append(x0)
    for i in range(numx):
      x1 = mu * x0 * (1 - x0)
      massive.append(x1)
      x0 = x1
    return np.array(massive)

def Xenon_Model(x0,x1,a,b, n):
    massive = []
    massive.append(x0)
    massive.append(x1)
    print(x0)
    print(x1)
    for i in range(n):
        x2 = 1 - a * x1**2 + b * x1
        massive.append(x2)
        x0 = x1
        x1 = x2
    return np.array(massive)

def TryToFindMC_2(full_series, k, ind_start, ind_end):
    matrix = np.array(full_series[ind_start:ind_end])
    tm.PCA_2(matrix, k)

def TryingWithRandowValue():#функция проверяет значения Cne|logn + e и тд на рандомном ряде
    CheckingTheProbabilityToAnalis(RandomSeries(0, 1, 10000), 5, 4, 10, 1, 1000, 2000)

def Try_sth(full_series, k, ind_start, ind_end):
    matrix = np.array(full_series[ind_start:ind_end])
    tm.Try_to_find_sth(matrix, k)

if __name__ == '__main__':
    #CheckingTheProbabilityToAnalis(FindFileFrom('Алюминий 2 серия.xlsx', 2, 2), 20, 0.005, 5500, 8500)
    #ex4('Алюминий 2 серия.xlsx', 2, 2, 20, 0.0009, 5500, 8500)
    #TryToFindMC(FindFileFrom('Алюминий 2 серия.xlsx', 2, 2), 5, 5500, 8500)
    #TryToFindMC(RandomSeries(10000), 5, 5500, 8500)
    #TryingWithRandowValue()
    #ex4(Logistic_Model(0.2,10000), 10, 1, 1, 0.001, 100, 1000)
    #CheckingTheProbabilityToAnalis(Logistic_Model(0.2,10000), 5, 4, 20, 1, 1000, 2000)
    #CheckingTheProbabilityToAnalis(Xenon_Model(-0.877, 0.257, 1.8, -0.005, 10000), 5, 4, 20, 1, 1000, 2000)
    #CheckingTheProbabilityToAnalis(Xenon_Model(-0.877, 0.257, 1.49, -0.138, 10000), 5, 4, 20, 1, 1000, 2000)
    #TryToFindMC_2(FindFileFrom('Алюминий 2 серия.xlsx', 2, 2), 10, 5500, 8500)
    Try_sth(FindFileFrom('Алюминий 2 серия.xlsx', 2, 2), 10, 5500, 8500)