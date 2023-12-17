import openpyxl
import math
from math import log
import os
import scipy
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


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

def FindTackensKoefInMatrix(name, n, e):
    matrix = []
    wb = openpyxl.load_workbook(filename=name)
    sheet = wb['Fract_4000']
    for j in range(2,331):
        tempEl = []
        for i in range(3,1623):
            tempEl.append(sheet.cell(row=i, column=j).value)
        matrix.append(tempEl)
    otv = FindC_n_e(matrix, n, e)
    print(otv)
    return 0