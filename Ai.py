import openpyxl
from sklearn import decomposition
import pandas as pd
def AI_Gen():
    keep_cols = [i for i in range(1,4)]
    df = pd.read_excel('Мед_сталь_микротрубка_15В.xlsx', sheet_name='Стальная_труб',skiprows=lambda x: x in [0, 1], usecols = keep_cols)
    pca = decomposition.PCA()
    pca.fit(df)
    pca.transform(df)
    #print(pca.transform(df))
    print(pca.explained_variance_ratio_)
    #print(df)

AI_Gen()


