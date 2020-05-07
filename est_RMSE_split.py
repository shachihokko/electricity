#########################################
#--------------- import -----------------
#########################################

import os
import pandas as pd
import numpy as np
import itertools
from dateutil.relativedelta import relativedelta
import datetime as dt
from tqdm import tqdm
import numba
from multiprocessing import Pool, freeze_support, RLock

#########################################
#--------- function setting ------------
#########################################

#変数作成のための係数の組み合わせ作成
def coef_vec_combi(num_vars):
  if num_vars <= 0:
    print("input Natural Number as num_vars")
    return None
  elif num_vars == 1:
    return [1]
  elif num_vars == 2:
    return [[1,1],[-1,1]]
  else:
    i = 2
    coef_vec = [[1,1],[-1,1]]
    while i < num_vars:
      i += 1
      coef_vec = [[1] + j for j in coef_vec] + [[-1] + j for j in coef_vec]
    return coef_vec

#変数作成のための変数の組み合わせ作成
def vars_combi(num_vars, col_names):
  return list(itertools.combinations(col_names,num_vars))

#numbaを用いた逆行列の高速化コード_2倍くらい早くなる
@numba.jit("f8[:,:](f8[:,:])")
def inv_nla_jit(A):
  return np.linalg.inv(A)

#予測誤差の2乗値の計算
#なお予測誤差は直近から1期前までのデータを用いて直近の実績値に対して評価
def eval_SE(Matrix_y, Matrix_x):
  #推計用データ
  train_x = Matrix_x[0:-1,:]
  train_y = Matrix_y[0:-1,:]
  #予測誤差評価データ
  test_x = Matrix_x[-1,:]
  test_y = Matrix_y[-1,:]
  #回帰係数行列の作成
  coef_reg = np.dot(inv_nla_jit(np.dot(train_x.T,train_x)),np.dot(train_x.T, train_y))
  return np.square(test_y-np.dot(test_x,coef_reg))

#並列化用に予測誤差の結果出力を関数化
def make_se_list(args):
  #ラップしてある引数から使用変数の取得
  mx_x_iip, mx_y = args[0], args[1]
  mx_elec, mx_const = args[2], args[3]
  coef_matrix, vc_list = args[4], args[5]
  row = len(mx_y)
  #予測誤差を算出
  SE = []
  for i in range(len(coef_matrix.T)):
    coeff = coef_matrix[:,i]
    SE = SE + [float(eval_SE(mx_y, np.concatenate([mx_const, mx_x_iip, np.dot(mx_elec[:, j], coeff).reshape([row,1])],1))) for j in vc_list]
  #バッチ毎の結果のまとめ
  return SE

#並列化計算用のラッパー(poolは引数１個しか渡せないのでラッパー噛ます)
def wrapper_make_se_list(args):
    return make_se_list(args)

#########################################
#--------------- setting ----------------
#########################################

#ファクター作成に使用する時間帯の数
num_vars = 6

#dir = os.path.dirname(os.path.abspath(__file__))
dir = "C:\\Users\\Desktop\\electricity\\"

#データ格納してるファイルの名前
file_name_iip = "data_iip"
file_name_elec = "data_elec"

#電力集計時期の上旬・上中旬・全部の調整
#data_上旬, data_上中旬, data, のどれかを選ぶ
elec_type = "data"

#時間帯別の列名の名前
col_names = [i for i in range(0,24)]

#推計（RMSE評価）に使用するデータの初期と終期
sdate = dt.datetime.strptime("2016-05-01", "%Y-%m-%d")
edate = dt.datetime.strptime("2020-02-01", "%Y-%m-%d")

#RMSE評価の開始時期
rmse_sdate = dt.datetime.strptime("2017-01-01", "%Y-%m-%d")

#説明変数の名前のリスト
x_name = ["iip_lag", "iip_f"]
#被説明変数の名前のリスト
y_name = ["iip"]

#########################################
#-------------- main process ------------
#########################################

#RMSE評価の回数の算出
dd = relativedelta(edate, rmse_sdate)
RMSE_loop = dd.years*12 + dd.months + 1

#データ読み込み
data_iip = pd.read_excel(dir+file_name_iip+".xlsx", sheet_name="data", index_col=0, parse_dates=True)
data_elec = pd.read_excel(dir+file_name_elec+".xlsx", sheet_name=elec_type, index_col=0, parse_dates=True)

#使用データの分離
data_x_iip = data_iip.loc[:, x_name]
data_y = data_iip.loc[:, y_name]

#データ期間の整理（これやらないと行列計算で困る）
data_x_iip = data_x_iip.loc[sdate:edate,:]
data_y = data_y.loc[sdate:edate,:]
data_elec = data_elec.loc[sdate:edate,:]

#定数項
mx_const = np.ones(len(data_x_iip), dtype="float64").reshape(len(data_x_iip),1)

#各種高速化のためにnumpy.ndarrayの形式に変換
mx_x_iip = np.array(data_x_iip)
mx_y = np.array(data_y)
mx_elec = np.array(data_elec)

### ファクター算出用の時間帯合成係数行列の作成（行列計算の関係上、転置）
coef_matrix = np.array(coef_vec_combi(num_vars)).T

### factorの作成
vc_list = vars_combi(num_vars, col_names)
num_batch = len(coef_matrix.T)

# 並列計算用に用いるコアの数
cores = os.cpu_count()

# 並列計算用に1コア当たりの仕事量に分割_//で整数部分の解のみを抽出
n_split = len(coef_matrix.T)//cores + 1
f_coef_list = []
for i in range(cores):
  if i != cores:
    f_coef_list.append(coef_matrix[:,n_split*i:n_split*(i+1)])
  #余りは最後のコアが全て受け取る
  elif i == cores:
    f_coef_list.append(coef_matrix[:,n_split*i:])


for RMSE_idx in range(RMSE_loop):
  ### 並列化計算のスタート
  if __name__ == "__main__":

    #RMSEのずらし
    if RMSE_idx!=0:
      mx_x_iip_tmp = mx_x_iip[0:-RMSE_idx,:]
      mx_y_tmp = mx_y[0:-RMSE_idx,:]
      mx_elec_tmp = mx_elec[0:-RMSE_idx,:]
      mx_const_tmp = mx_const[0:-RMSE_idx,:]
    elif RMSE_idx==0:
      mx_x_iip_tmp = mx_x_iip
      mx_y_tmp = mx_y
      mx_elec_tmp = mx_elec
      mx_const_tmp = mx_const

    #並列化時の引数の作成
    #coefの組み合わせで分割させる
    wrap_multi_args =[(mx_x_iip_tmp, mx_y_tmp, mx_elec_tmp, mx_const_tmp, f_coef_list[i], vc_list) for i in range(cores)]

    #並列化処理
    with Pool(processes=os.cpu_count()) as p:
      result = list(tqdm(p.imap(wrapper_make_se_list, wrap_multi_args), total = cores))

    result = sum(result,[])

    #出力
    output = pd.DataFrame()
    output["RMSE"] = result
    output.to_csv(dir+"split\\"+"RMSE_"+str(num_vars)+"_"+str(RMSE_idx)+".csv.bz2", compression='bz2', encoding="cp932")
