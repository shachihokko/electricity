import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import itertools

#dir = os.path.dirname(os.path.abspath(__file__))
dir = "C:\\Users\\Desktop\\electricity\\"
file_name = "data"
data = pd.read_excel(dir+file_name+".xlsx", sheet_name="data", index_col=0, parse_dates=True)

data_x = data.loc["2016-05-01":"2020-02-01",["iip_lag", "iip_f", "h_all"]]
data_y = data.loc["2016-05-01":"2020-02-01",["iip"]]

train_x = data_x.loc["2016-05-01":"2019-12-01",:]
train_y = data_y.loc["2016-05-01":"2019-12-01",:]

test_x = data_x.loc["2020-01-01":"2020-01-01",:]
test_y = data_y.loc["2020-01-01":"2020-01-01",:]



combi = range(-1, 2)  # -1, 0, 1
combi_list = []
for i in itertools.product(origin_list, repeat=4):
  combi_list.append(i)


#モデルの指定
model = LinearRegression()
#モデルの訓練
model.fit(train_x, train_y)
#モデルによる予測値の表示
model.predict(test_x)


