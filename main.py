import openpyxl
import math
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import TackensMethof as tm
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def FindFileFrom(name,x ,y):
    wb = openpyxl.load_workbook(filename=name)
    sheet = wb['1']
    time_series, time_array = [], []
    while sheet.cell(row=x, column=y).value != None:
        time_series.append(float(sheet.cell(row=x, column=y).value))
        time_array.append(float(sheet.cell(row=x, column=y - 1).value))
        x += 1
    return np.array(time_series)

def CheckingTheProbabilityToAnalis(matrix, n, numn, nume, epsilon, ind_start, ind_end):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(projection="3d")
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

def Logistic_Model(x0,mu, numx):
    #mu = 4
    massive = [x0]
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
        x2 = 1 - a * x1**2 + b * x0
        massive.append(x2)
        x0 = x1
        x1 = x2
    return np.array(massive)

def TryToFindMC_2(full_series, k, ind_start, ind_end):
    matrix = np.array(full_series[ind_start:ind_end])
    tm.PCA_2(matrix, k)

def TryingWithRandowValue():#функция проверяет значения Cne|logn + e и тд на рандомном ряде
    CheckingTheProbabilityToAnalis(RandomSeries(0, 1, 10000), 5, 4, 10, 1, 1000, 2000)

def MakePlotsOfComponentsValue(full_series, k, how_many_k_show, ind_start, ind_end):
    series = np.array(full_series[ind_start:ind_end])
    vector_projection = tm.MakingArrayOfComponentsValue_2(series, k)
    for i in range(how_many_k_show):
        plt.plot(vector_projection[i], label=f"Компонента = {i}")
        plt.legend()
        plt.show()

def Make3DCloudOfPoints(number_of_points):
    return np.random.random(number_of_points)

def ExperimentIn3Dimension(number_of_points):
    x = Make3DCloudOfPoints(number_of_points)
    y = x * 2 + 1 + np.random.random(number_of_points) * 3
    z = 2 * x + 3 * y + np.random.random(number_of_points) * 6
    matrix = np.dstack((x, y, z))
    centered_matrix = np.dstack((x - x.mean(), y - y.mean(), z - z.mean()))
    covmat = np.cov((x - x.mean(), y - y.mean(), z - z.mean()))
    print(covmat)
    eigenvalues, eigenvectors = LA.eig(covmat)
    print(eigenvectors)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.plot([np.mean(x), np.mean(x) + eigenvectors[0][0]], [np.mean(y), np.mean(y) + eigenvectors[1][0]],
            zs=[np.mean(z), np.mean(z) + eigenvectors[2][0]], color='green')
    ax.plot([np.mean(x), np.mean(x) + eigenvectors[0][1]], [np.mean(y), np.mean(y) + eigenvectors[1][1]],
            zs=[np.mean(z), np.mean(z) + eigenvectors[2][1]], color='orange')
    ax.plot([np.mean(x), np.mean(x) + eigenvectors[0][2]], [np.mean(y), np.mean(y) + eigenvectors[1][2]],
            zs=[np.mean(z), np.mean(z) + eigenvectors[2][2]], color='purple')
    plt.show()
    return 0
if __name__ == '__main__':
    #CheckingTheProbabilityToAnalis(FindFileFrom('Алюминий 2 серия.xlsx', 2, 2), 20, 0.005, 5500, 8500)
    #ex4('Алюминий 2 серия.xlsx', 2, 2, 20, 0.0009, 5500, 8500)
    #TryToFindMC(FindFileFrom('Алюминий 2 серия.xlsx', 2, 2), 5, 5500, 8500)
    #TryToFindMC(RandomSeries(10000), 5, 5500, 8500)
    #TryingWithRandowValue()
    #ex4(Logistic_Model(0.2,10000), 10, 1, 1, 0.001, 100, 1000)
    #CheckingTheProbabilityToAnalis(Logistic_Model(0.2, 4, 10000), 3, 2, 20, 1, 1000, 2000)
    #CheckingTheProbabilityToAnalis(Xenon_Model(-0.877, 0.257, 1.8, -0.005, 10000), 5, 4, 20, 1, 1000, 2000)
    #CheckingTheProbabilityToAnalis(Xenon_Model(-0.877, 0.257, 1.49, -0.138, 10000), 5, 4, 20, 1, 1000, 2000)
    #TryToFindMC_2(FindFileFrom('Алюминий 2 серия.xlsx', 2, 2), 10, 5500, 8500)
    MakePlotsOfComponentsValue(FindFileFrom('Алюминий 2 серия.xlsx', 2, 2), 300, 1, 5500, 8500)
    #ExperimentIn3Dimension(100)