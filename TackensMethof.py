import openpyxl
import math
from math import log
import os
import scipy
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


def HowManyElementsInG(vector, n, e):
    g = 0
    for i in range(1, len(vector) - n):
        i_is_in_G = True
        for j in range(i):
            temp_max = abs(vector[i] - vector[j])
            for k in range(1,n+1):
                if(abs(vector[i + k] - vector[j + k]) > temp_max):
                    temp_max = abs(vector[i + k] - vector[j + k])
            if(temp_max < e):
                i_is_in_G = False
        if(i_is_in_G is True):
            g += 1
    return g
def FindCInVector(vector, n, e):
    temp_el = []
    for j in range(len(vector)):
        temp_el.append(vector[j])
    C_n_e = HowManyElementsInG(temp_el, n, e)
    if (C_n_e == 0):
        print("C = 0")
        otv = 0
    else:
        otv = log(C_n_e) / (n - log(e))
    return otv
def FindC_n_e(matrix, n, e):
    rezult_vector = []
    for i in range(len(matrix[0])):
        temp_el = []
        for j in range(len(matrix)):
            temp_el.append(matrix[j][i])
        C_n_e = HowManyElementsInG(temp_el, n, e)
        if (C_n_e == 0):
            print("C = 0")
            otv = 0
        else:
            otv = log(C_n_e) / (n - log(e))
        rezult_vector.append(otv)
    return rezult_vector
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