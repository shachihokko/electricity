
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os


#dir = os.path.dirname(os.path.abspath(__file__))
dir = "C:\\Users\\Desktop\\electricity\\electricity\\"
X13PATH = "C:\\WinX13\\x13as"
file_name = "monthly_九州"
data = pd.read_csv(dir+file_name+".csv", engine="python", index_col=0, parse_dates=True)


#何でかわからんが、pathが通ってない？
#作業ディレクトリをX13直下にすると動く
os.chdir(X13PATH)


col_list = data.columns
idx_list = data.index
output=pd.DataFrame()
for i in col_list:
  x13results = sm.tsa.x13_arima_analysis(endog = data[i], outlier=True, freq="12")
  output[i] = list(x13results.seasadj)
output.index = idx_list
output.to_csv(dir+file_name+"_sa.csv", encoding="cp932")
