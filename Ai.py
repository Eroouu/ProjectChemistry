import openpyxl
from sklearn import decomposition
import pandas as pd
def AI_Gen():
    keep_cols = [i for i in range(1,67)]
    df = pd.read_excel('Медь_толстая_проволока№1_15В.xlsx', sheet_name='Cu1',skiprows=lambda x: x in [0, 1], usecols = keep_cols)
    df.columns = df.columns.astype(str)
    pca = decomposition.PCA()
    pca.fit(df)
    pca.transform(df)
    #print(pca.transform(df))
    print(pca.explained_variance_ratio_)
    #print(df)

AI_Gen()


