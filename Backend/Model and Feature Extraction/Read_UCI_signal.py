# @author : Yulin Huang
# @email : yulinhuang0217@gmail.com
# @software: pycharm
import numpy as np
import pandas as pd

def read_file(Path):
    """Read the .txt, .data or .csv file as dataframe. Input a parameter 'Path', which indicates the location of the file."""
    dataframe = pd.read_csv(filepath_or_buffer=Path, header=None, delim_whitespace=True)
    return dataframe

def get_array(dataframe):
    X = np.array(dataframe)
    return X

#For training
body_x = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt')
body_x_arr = get_array(body_x)
body_y = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/body_acc_y_train.txt')
body_y_arr = get_array(body_y)
body_z = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/body_acc_z_train.txt')
body_z_arr = get_array(body_z)
#Merge
body = np.hstack((body_x_arr,body_y_arr,body_z_arr))


total_acc_x = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt')
total_acc_x_arr = get_array(total_acc_x)
total_acc_y = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt')
total_acc_y_arr = get_array(total_acc_y)
total_acc_z = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt')
total_acc_z_arr = get_array(total_acc_z)
#Merge
total_acc = np.hstack((total_acc_x_arr,total_acc_y_arr,total_acc_z_arr))

gry_x = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt')
gry_x_arr = get_array(gry_x)
gry_y = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/body_gyro_y_train.txt')
gry_y_arr = get_array(gry_y)
gry_z = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/body_gyro_z_train.txt')
gry_z_arr = get_array(gry_z)

#平均绝对偏差
mad = []
for i in range(0,len(gry_x_arr)):
    series = pd.Series(gry_x_arr[i])  # 利用Series将列表转换成新的、pandas可处理的数据
    mad.append(series.mad())

#相关系数
"""corr_gust= []
for i in range(0,len(gyro_x_arr)):
    g_s_m = pd.Series(gyro_x_arr[i])  # 利用Series将列表转换成新的、pandas可处理的数据
    g_a_d = pd.Series(gyro_y_arr[i])
    corr_gust.append(round(g_s_m.corr(g_a_d), 4))  # 计算标准差，round(a, 4)是保留a的前四位小数
print(corr_gust)"""

#IQR四分位差

"""int_r = []
for i in range(0,len(gyro_x_arr)):
    lower_q=np.quantile(gyro_x_arr[i],0.25,method='lower')#下四分位数
    higher_q=np.quantile(gyro_y_arr[i],0.75,method='higher')#上四分位数
    int_r.append(higher_q-lower_q)
print(int_r)"""

#Merge
gry = np.hstack((gry_x_arr,gry_y_arr,gry_z_arr))

final = np.hstack((body,gry))

#print(final)

def return_final_train():
    """Return final merged train signals"""
    return final


