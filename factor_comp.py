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

#並列化計算用のラッパー(poolは引数１個しか渡せないのでラッパー噛ます)
def wrapper(args):
    return make_factor_composition(args)

def make_factor_composition(args):
  #ラップしてある引数から使用変数の取得
  n, coef_matrix, vc_list = args[0], args[1], args[2]
  # 進捗バーの左側に表示される文字列
  info = f'#{n:>2} '

  num_batch = len(coef_matrix.T)
  factor_table = np.zeros([num_batch*len(vc_list),24])
  for i in tqdm(range(num_batch), desc=info, position=n+1):
    coeff = coef_matrix[:,i]
    count = 0
    for j in vc_list:
      factor_table[i*len(vc_list)+count, j] = coeff
      count += 1
  return factor_table

#########################################
#--------------- setting ----------------
#########################################

#ファクター作成に使用する時間帯の数
num_vars = 5

#dir = os.path.dirname(os.path.abspath(__file__))
dir = "C:\\Users\\tomo\\Desktop\\electricity\\"

hour_names = ["h{0:02d}".format(i)+"_"+"{0:02d}".format(i+1) for i in range(0,24)]
col_names = [i for i in range(0,24)]


#########################################
#-------------- main process ------------
#########################################

### ファクター算出用の時間帯合成係数行列の作成（行列計算の関係上、転置）
coef_matrix = np.array(coef_vec_combi(num_vars)).T

### 対応表の作成

# 要素の組み合わせ取得
vc_list = vars_combi(num_vars, col_names)
# 並列計算用に用いるコアの数
cores = os.cpu_count()

# 並列計算するかの分岐
if cores > len(coef_matrix.T):
  num_batch = len(coef_matrix.T)
  factor_table = np.zeros([num_batch*len(vc_list),24])
  for i in tqdm(range(num_batch)):
    coeff = coef_matrix[:,i]
    count = 0
    for j in vc_list:
      factor_table[i*len(vc_list)+count, j] = coeff
      count += 1
  #出力
  factor_table = pd.DataFrame(factor_table, columns=hour_names)
  factor_table.to_csv(dir+"table_f"+str(num_vars)+".csv.bz2", compression='bz2', encoding="cp932")

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
    wrap_multi_args =[(i, f_coef_list[i], vc_list) for i in range(os.cpu_count())]

    #並列化処理
    with Pool(processes=os.cpu_count(), initializer=tqdm.set_lock, initargs=(RLock(),)) as p:
      result = p.map(wrapper, wrap_multi_args)

    # tqdm終了後のカーソル位置を最下部に持ってくる
    print("\n" * os.cpu_count())

    #出力
    factor_table = np.array(result).flatten()
    factor_table = factor_table.reshape([int(len(factor_table)/24),24])
    factor_table = pd.DataFrame(factor_table, columns=hour_names)
    factor_table.to_csv(dir+"table_f"+str(num_vars)+".csv.bz2", compression='bz2', encoding="cp932")
