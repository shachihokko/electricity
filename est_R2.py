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

#R2
def R2_calc(Matrix_y, Matrix_x, R2_denominator):
  coef_reg = np.dot(inv_nla_jit(np.dot(Matrix_x.T,Matrix_x)),np.dot(Matrix_x.T, Matrix_y))
  R2 = 1-(np.sum(np.square(Matrix_y-np.dot(Matrix_x,coef_reg))))/R2_denominator
  return float(R2)

#並列化用に予測誤差の結果出力を関数化
def make_R2_list(args):
  #ラップしてある引数から使用変数の取得
  n, r2_denom = args[0], args[1]
  mx_x_iip, mx_y = args[2], args[3]
  mx_elec, mx_const = args[4], args[5]
  coef_matrix, vc_list = args[6], args[7]
  # 進捗バーの左側に表示される文字列
  info = f'#{n:>2} '
  #reshape用
  row = len(mx_y)
  #R2を算出
  R2 = []
  for i in tqdm(range(len(coef_matrix.T)), desc=info, position=n+1):
    coeff = coef_matrix[:,i]
    R2 = R2 + [R2_calc(mx_y, np.concatenate([mx_const, mx_x_iip,np.dot(mx_elec[:, j], coeff).reshape([row,1])],1),r2_denom) for j in vc_list]
  #バッチ毎の結果のまとめ
  return R2

#並列化計算用のラッパー(poolは引数１個しか渡せないのでラッパー噛ます)
def wrapper(args):
    return make_R2_list(args)

#########################################
#--------------- setting ----------------
#########################################

#ファクター作成に使用する時間帯の数
num_vars = 2

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

#説明変数の名前のリスト
x_name = ["iip_lag", "iip_f"]
#被説明変数の名前のリスト
y_name = ["iip"]

#########################################
#-------------- main process ------------
#########################################

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

#R2の計算用にyの平均と実績の乖離の２乗値のベクトル作成
mx_y_mean = np.mean(mx_y)*np.ones(len(data_y), dtype="float64").reshape(len(mx_y),1)
r2_denom = np.sum(np.square(mx_y-mx_y_mean))

### ファクター算出用の時間帯合成係数行列の作成（行列計算の関係上、転置）
coef_matrix = np.array(coef_vec_combi(num_vars)).T

### factorの作成

# 要素の組み合わせ取得
vc_list = vars_combi(num_vars, col_names)
# 並列計算用に用いるコアの数
cores = os.cpu_count()

# 並列計算するかの分岐
if cores > len(coef_matrix.T):
  #R2を算出
  row = len(mx_y)
  R2 = []
  for i in tqdm(range(len(coef_matrix.T))):
    coeff = coef_matrix[:,i]
    R2 = R2 + [R2_calc(mx_y, np.concatenate([mx_const, mx_x_iip,np.dot(mx_elec[:, j], coeff).reshape([row,1])],1),r2_denom) for j in vc_list]

  #出力
  output = pd.DataFrame()
  output["R2"]=R2
  output.to_csv(dir+"R2_para"+str(num_vars)+".csv.bz2", compression='bz2', encoding="cp932")

else:
  # 並列計算用に1コア当たりの仕事量に分割_//で整数部分の解のみを抽出
  n_split = len(coef_matrix.T)//cores
  f_coef_list = []
  for i in range(cores):
    if i != cores:
      f_coef_list.append(coef_matrix[:,n_split*i:n_split*(i+1)])
    #余りは最後のコアが全て受け取る
    elif i == cores:
      f_coef_list.append(coef_matrix[:,n_split*i:])

  ### 並列化計算のスタート
  if __name__ == "__main__":

    #tqdmの表示のためにWindowsでは必要
    #__name__=="__main__"の直後に宣言する必要あり
    freeze_support()

    #並列化時の引数の作成
    wrap_multi_args =[(i, r2_denom, mx_x_iip, mx_y, mx_elec, mx_const, f_coef_list[i], vc_list) for i in range(os.cpu_count())]

    #並列化処理
    with Pool(processes=os.cpu_count(), initializer=tqdm.set_lock, initargs=(RLock(),)) as p:
      result = p.map(wrapper, wrap_multi_args)

    # tqdm終了後のカーソル位置を最下部に持ってくる
    print("\n" * os.cpu_count())

    #出力
    result = np.array(result)
    result = result.reshape([1,len(result)*len(result.T)]).tolist()[0]
    output = pd.DataFrame()
    output["R2"]=result
    output.to_csv(dir+"R2_para"+str(num_vars)+".csv.bz2", compression='bz2', encoding="cp932")
