
import pandas as pd
import numpy as np
import datetime as dt
import os

#dir = os.path.dirname(os.path.abspath(__file__)) + "\\"
dir = "C:\\Users\\Desktop\\electricity\\electricity\\"
data = pd.read_csv(dir+"電力需要実績.csv", engine="python")

#抽出対象（"上旬"、 "上中旬"、 ""<全部>）から選択
key_range = ""

#抽出対象
#key_region = "１０エリア計"
key_region = "九州"

#時間キーの準備
key_hour_list = []
col_name = []
for hh in range(0,24):
  key_hour_list.append("{0:02d}".format(hh) + ":00～" + "{0:02d}".format(hh+1) + ":00")
  col_name.append("h"+"{0:02d}".format(hh)+"_"+"{0:02d}".format(hh+1))
key_hour_list.append("日電力量(MWh)")
col_name.append("h_all")

#上中旬抽出用に日付の型変換
data["年月日"] = pd.to_datetime(data["年月日"], format="%Y/%m/%d")

#出力用のリスト準備
elec_list = []
idx = []

for yy in range(2016,2021):
  for mm in range(1,13):
    date = "{0:04d}".format(yy) + "{0:02d}".format(mm)
    key_date = "{0:04d}".format(yy) + "/" + "{0:02d}".format(mm) + "/"

    #スキップ処理
    if int(date) < int("201604"):
      continue
    elif int(date) > int("202003"):
      break

    #インデックスの作成
    idx.append(dt.datetime.strptime("{0:04d}".format(yy) + "-" + "{0:02d}".format(mm), "%Y-%m"))

    for key_hour in key_hour_list:

      #時間帯の抽出
      data_tmp = data[data["時間帯"]==key_hour]

      #年の抽出
      data_tmp = data_tmp[data_tmp["年月日"].dt.year == yy]
      #月の抽出
      data_tmp = data_tmp[data_tmp["年月日"].dt.month == mm]
      #日の抽出
      if key_range == "上旬":
        data_tmp = data_tmp[data_tmp["年月日"].dt.day <= 10]
      elif key_range == "上中旬":
        data_tmp = data_tmp[data_tmp["年月日"].dt.day <= 20]

      #対象の月・時間帯の電力需要量の抽出
      #1*nmのリストになってるので後でn*mに修正する
      elec_list.append(data_tmp[key_region].sum())

elec = np.array(elec_list)
elec = elec.reshape([len(idx),len(key_hour_list)])
elec = pd.DataFrame(elec, columns=col_name, index=idx)

output_path = dir + "monthly" + "_" + key_region + key_range +".csv"
elec.to_csv(output_path, encoding="cp932")
