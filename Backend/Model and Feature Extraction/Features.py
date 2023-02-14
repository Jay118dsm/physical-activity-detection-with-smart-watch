# @author : Yulin Huang
# @email : yulinhuang0217@gmail.com
# @software: pycharm
import pandas as pd
import numpy as np
import Data_Input
import warnings
import joblib
import pandas as pd
import Read_test_signal_csv

# 忽略警告
warnings.filterwarnings("ignore")


def read_file(Path):
    """Read the .txt, .data or .csv file as dataframe. Input a parameter 'Path', which indicates the location of the file."""
    dataframe = pd.read_csv(filepath_or_buffer=Path, header=None, delim_whitespace=True)
    return dataframe


def get_array(dataframe):
    X = np.array(dataframe)
    return X


# For training
body_x = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt')
body_x_arr = get_array(body_x)
body_y = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/body_acc_y_train.txt')
body_y_arr = get_array(body_y)
body_z = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/body_acc_z_train.txt')
body_z_arr = get_array(body_z)
# Merge
body = np.hstack((body_x_arr, body_y_arr, body_z_arr))

total_acc_x = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt')
total_acc_x_arr = get_array(total_acc_x)
total_acc_y = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt')
total_acc_y_arr = get_array(total_acc_y)
total_acc_z = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt')
total_acc_z_arr = get_array(total_acc_z)
# Merge
total_acc = np.hstack((total_acc_x_arr, total_acc_y_arr, total_acc_z_arr))

gyro_x = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt')
gyro_x_arr = get_array(gyro_x)
gyro_y = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/body_gyro_y_train.txt')
gyro_y_arr = get_array(gyro_y)
gyro_z = read_file('Data Set/UCI HAR Dataset/train/Inertial Signals/body_gyro_z_train.txt')
gyro_z_arr = get_array(gyro_z)
gyro = np.hstack((gyro_x_arr, gyro_y_arr, gyro_z_arr))

# For testing(F)
test_body_x = Read_test_signal_csv.read_test_file('finish_data/Merge/body_xAcc_final.csv')
test_body_x_arr = get_array(test_body_x)
test_body_y = Read_test_signal_csv.read_test_file('finish_data/Merge/body_yAcc_final.csv')
test_body_y_arr = get_array(test_body_y)
test_body_z = Read_test_signal_csv.read_test_file('finish_data/Merge/body_zAcc_final.csv')
test_body_z_arr = get_array(test_body_z)
# Merge
test_body = np.hstack((test_body_x_arr, test_body_y_arr, test_body_z_arr))

test_total_acc_x = Read_test_signal_csv.read_test_file('finish_data/Merge/gra_xAcc_final.csv')
test_total_acc_x_arr = get_array(test_total_acc_x)
test_total_acc_y = Read_test_signal_csv.read_test_file('finish_data/Merge/gra_yAcc_final.csv')
test_total_acc_y_arr = get_array(test_total_acc_y)
test_total_acc_z = Read_test_signal_csv.read_test_file('finish_data/Merge/gra_zAcc_final.csv')
test_total_acc_z_arr = get_array(test_total_acc_z)
# Merge
test_total_acc = np.hstack((test_total_acc_x_arr, test_total_acc_y_arr, test_total_acc_z_arr))

test_gyro_x = Read_test_signal_csv.read_test_file('finish_data/Merge/xGyr_final.csv')
test_gyro_x_arr = get_array(test_gyro_x)
test_gyro_y = Read_test_signal_csv.read_test_file('finish_data/Merge/yGyr_final.csv')
test_gyro_y_arr = get_array(test_gyro_y)
test_gyro_z = Read_test_signal_csv.read_test_file('finish_data/Merge/zGyr_final.csv')
test_gyro_z_arr = get_array(test_gyro_z)
test_gyro = np.hstack((test_gyro_x_arr, test_gyro_y_arr, test_gyro_z_arr))

# For testing(UCI)
UCI_test_body_x = read_file('Data Set/UCI HAR Dataset/test/Inertial Signals/body_acc_x_test.txt')
UCI_test_body_x_arr = get_array(UCI_test_body_x)
UCI_test_body_y = read_file('Data Set/UCI HAR Dataset/test/Inertial Signals/body_acc_y_test.txt')
UCI_test_body_y_arr = get_array(UCI_test_body_y)
UCI_test_body_z = read_file('Data Set/UCI HAR Dataset/test/Inertial Signals/body_acc_z_test.txt')
UCI_test_body_z_arr = get_array(UCI_test_body_z)
# Merge
UCI_test_body = np.hstack((UCI_test_body_x_arr, UCI_test_body_y_arr, UCI_test_body_z_arr))

UCI_test_total_acc_x = read_file('Data Set/UCI HAR Dataset/test/Inertial Signals/total_acc_x_test.txt')
UCI_test_total_acc_x_arr = get_array(UCI_test_total_acc_x)
UCI_test_total_acc_y = read_file('Data Set/UCI HAR Dataset/test/Inertial Signals/total_acc_y_test.txt')
UCI_test_total_acc_y_arr = get_array(UCI_test_total_acc_y)
UCI_test_total_acc_z = read_file('Data Set/UCI HAR Dataset/test/Inertial Signals/total_acc_z_test.txt')
UCI_test_total_acc_z_arr = get_array(UCI_test_total_acc_z)
# Merge
UCI_test_total_acc = np.hstack((UCI_test_total_acc_x_arr, UCI_test_total_acc_y_arr, UCI_test_total_acc_z_arr))

UCI_test_gyro_x = read_file('Data Set/UCI HAR Dataset/test/Inertial Signals/body_gyro_x_test.txt')
UCI_test_gyro_x_arr = get_array(UCI_test_gyro_x)
UCI_test_gyro_y = read_file('Data Set/UCI HAR Dataset/test/Inertial Signals/body_gyro_y_test.txt')
UCI_test_gyro_y_arr = get_array(UCI_test_gyro_y)
UCI_test_gyro_z = read_file('Data Set/UCI HAR Dataset/test/Inertial Signals/body_gyro_z_test.txt')
UCI_test_gyro_z_arr = get_array(UCI_test_gyro_z)
UCI_test_gyro = np.hstack((UCI_test_gyro_x_arr, UCI_test_gyro_y_arr, UCI_test_gyro_z_arr))


# 输入9个基础惯性数据，返回由十一终特征组成的ndarray，大小为99维。total_acc即为gra_acc
def features_extraction(gyro_x_arr, gyro_y_arr, gyro_z_arr, body_x_arr, body_y_arr, body_z_arr, total_acc_x_arr,
                        total_acc_y_arr, total_acc_z_arr):
    """输入9个基础惯性数据，返回由十一终特征组成的ndarray，大小为99维。"""
    # 特征提取
    # 1,相关系数
    corr_gust_gyro_xy = []
    corr_gust_gyro_xz = []
    corr_gust_gyro_yz = []
    corr_gust_gra_xy = []
    corr_gust_gra_xz = []
    corr_gust_gra_yz = []
    corr_gust_body_xy = []
    corr_gust_body_xz = []
    corr_gust_body_yz = []
    for i in range(0, len(gyro_x_arr)):
        x_gry = pd.Series(gyro_x_arr[i])  # 利用Series将列表转换成新的、pandas可处理的数据
        y_gry = pd.Series(gyro_y_arr[i])
        z_gry = pd.Series(gyro_z_arr[i])
        x_body = pd.Series(body_x_arr[i])
        y_body = pd.Series(body_y_arr[i])
        z_body = pd.Series(body_z_arr[i])
        x_gra = pd.Series(total_acc_x_arr[i])
        y_gra = pd.Series(total_acc_y_arr[i])
        z_gra = pd.Series(total_acc_z_arr[i])
        corr_gust_gyro_xy.append(round(x_gry.corr(y_gry), 8))
        corr_gust_gyro_xz.append(round(x_gry.corr(z_gry), 8))
        corr_gust_gyro_yz.append(round(y_gry.corr(z_gry), 8))  # 计算标准差，round(a, 4)是保留a的前四位小数
        corr_gust_gra_xy.append(round(x_gra.corr(y_gra), 8))
        corr_gust_gra_xz.append(round(x_gra.corr(z_gra), 8))
        corr_gust_gra_yz.append(round(y_gra.corr(z_gra), 8))
        corr_gust_body_xy.append(round(x_body.corr(y_body), 8))
        corr_gust_body_xz.append(round(x_body.corr(z_body), 8))
        corr_gust_body_yz.append(round(y_body.corr(z_body), 8))
    corr_gust = np.vstack((np.array(corr_gust_gyro_xy), np.array(corr_gust_gyro_xz), np.array(corr_gust_gyro_yz),
                           np.array(corr_gust_gra_xy), np.array(corr_gust_gra_xz), np.array(corr_gust_gra_yz),
                           np.array(corr_gust_body_xy), np.array(corr_gust_body_xz), np.array(corr_gust_body_yz))).T
    
    # 2,平均绝对偏差
    mad_gyro_x = []
    mad_gyro_y = []
    mad_gyro_z = []
    mad_body_x = []
    mad_body_y = []
    mad_body_z = []
    mad_gra_x = []
    mad_gra_y = []
    mad_gra_z = []
    for i in range(0, len(gyro_x_arr)):
        series1 = pd.Series(gyro_x_arr[i])
        mad_gyro_x.append(series1.mad())
        series2 = pd.Series(gyro_y_arr[i])
        mad_gyro_y.append(series2.mad())
        series3 = pd.Series(gyro_z_arr[i])
        mad_gyro_z.append(series3.mad())
        series4 = pd.Series(body_x_arr[i])
        mad_body_x.append(series4.mad())
        series5 = pd.Series(body_y_arr[i])
        mad_body_y.append(series5.mad())
        series6 = pd.Series(body_z_arr[i])
        mad_body_z.append(series6.mad())
        series7 = pd.Series(total_acc_x_arr[i])
        mad_gra_x.append(series7.mad())
        series8 = pd.Series(total_acc_y_arr[i])
        mad_gra_y.append(series8.mad())
        series9 = pd.Series(total_acc_z_arr[i])
        mad_gra_z.append(series9.mad())
    mad = np.vstack((np.array(mad_gyro_x), np.array(mad_gyro_y), np.array(mad_gyro_z), np.array(mad_body_x),
                     np.array(mad_body_y), np.array(mad_body_z), np.array(mad_gra_x), np.array(mad_gra_y),
                     np.array(mad_gra_z))).T
    
    # 3,IQR四分位差
    IQR_gyro_x = []
    IQR_gyro_y = []
    IQR_gyro_z = []
    IQR_body_x = []
    IQR_body_y = []
    IQR_body_z = []
    IQR_gra_x = []
    IQR_gra_y = []
    IQR_gra_z = []
    for i in range(0, len(gyro_x_arr)):
        lower_q1 = np.quantile(gyro_x_arr[i], 0.25, method='lower')  # 下四分位数
        higher_q1 = np.quantile(gyro_x_arr[i], 0.75, method='higher')  # 上四分位数
        lower_q2 = np.quantile(gyro_y_arr[i], 0.25, method='lower')  # 下四分位数
        higher_q2 = np.quantile(gyro_y_arr[i], 0.75, method='higher')  # 上四分位数
        lower_q3 = np.quantile(gyro_z_arr[i], 0.25, method='lower')  # 下四分位数
        higher_q3 = np.quantile(gyro_z_arr[i], 0.75, method='higher')  # 上四分位数
        IQR_gyro_x.append(higher_q1 - lower_q1)
        IQR_gyro_y.append(higher_q2 - lower_q2)
        IQR_gyro_z.append(higher_q3 - lower_q3)
        
        lower_q4 = np.quantile(body_x_arr[i], 0.25, method='lower')  # 下四分位数
        higher_q4 = np.quantile(body_x_arr[i], 0.75, method='higher')  # 上四分位数
        lower_q5 = np.quantile(body_y_arr[i], 0.25, method='lower')  # 下四分位数
        higher_q5 = np.quantile(body_y_arr[i], 0.75, method='higher')  # 上四分位数
        lower_q6 = np.quantile(body_z_arr[i], 0.25, method='lower')  # 下四分位数
        higher_q6 = np.quantile(body_z_arr[i], 0.75, method='higher')  # 上四分位数
        IQR_body_x.append(higher_q4 - lower_q4)
        IQR_body_y.append(higher_q5 - lower_q5)
        IQR_body_z.append(higher_q6 - lower_q6)
        
        lower_q7 = np.quantile(total_acc_x_arr[i], 0.25, method='lower')  # 下四分位数
        higher_q7 = np.quantile(total_acc_x_arr[i], 0.75, method='higher')  # 上四分位数
        lower_q8 = np.quantile(total_acc_y_arr[i], 0.25, method='lower')  # 下四分位数
        higher_q8 = np.quantile(total_acc_y_arr[i], 0.75, method='higher')  # 上四分位数
        lower_q9 = np.quantile(total_acc_z_arr[i], 0.25, method='lower')  # 下四分位数
        higher_q9 = np.quantile(total_acc_z_arr[i], 0.75, method='higher')  # 上四分位数
        IQR_gra_x.append(higher_q7 - lower_q7)
        IQR_gra_y.append(higher_q8 - lower_q8)
        IQR_gra_z.append(higher_q9 - lower_q9)
    IQR = np.vstack((np.array(IQR_gyro_x), np.array(IQR_gyro_y), np.array(IQR_gyro_z), np.array(IQR_body_x),
                     np.array(IQR_body_y), np.array(IQR_body_z), np.array(IQR_gra_x), np.array(IQR_gra_y),
                     np.array(IQR_gra_z))).T
    # 4, mean均值
    mean = np.vstack((np.mean(gyro_x_arr, axis=1), np.mean(gyro_y_arr, axis=1), np.mean(gyro_z_arr, axis=1),
                      np.mean(body_x_arr, axis=1), np.mean(body_y_arr, axis=1), np.mean(body_z_arr, axis=1),
                      np.mean(total_acc_x_arr, axis=1), np.mean(total_acc_y_arr, axis=1),
                      np.mean(total_acc_z_arr, axis=1))).T
    # 5, mid中位数
    median = np.vstack((np.median(gyro_x_arr, axis=1), np.median(gyro_y_arr, axis=1), np.median(gyro_z_arr, axis=1),
                        np.median(body_x_arr, axis=1), np.median(body_y_arr, axis=1), np.median(body_z_arr, axis=1),
                        np.median(total_acc_x_arr, axis=1), np.median(total_acc_y_arr, axis=1),
                        np.median(total_acc_z_arr, axis=1))).T
    # 6,max最大值
    max = np.vstack((np.max(gyro_x_arr, axis=1), np.max(gyro_y_arr, axis=1), np.max(gyro_z_arr, axis=1),
                     np.max(body_x_arr, axis=1), np.max(body_y_arr, axis=1), np.max(body_z_arr, axis=1),
                     np.max(total_acc_x_arr, axis=1), np.max(total_acc_y_arr, axis=1),
                     np.max(total_acc_z_arr, axis=1))).T
    # 7,min最小值
    min = np.vstack((np.min(gyro_x_arr, axis=1), np.min(gyro_y_arr, axis=1), np.min(gyro_z_arr, axis=1),
                     np.min(body_x_arr, axis=1), np.min(body_y_arr, axis=1), np.min(body_z_arr, axis=1),
                     np.min(total_acc_x_arr, axis=1), np.min(total_acc_y_arr, axis=1),
                     np.min(total_acc_z_arr, axis=1))).T
    # 8,std标准差
    std = np.vstack((np.std(gyro_x_arr, axis=1), np.std(gyro_y_arr, axis=1), np.std(gyro_z_arr, axis=1),
                     np.std(body_x_arr, axis=1), np.std(body_y_arr, axis=1), np.std(body_z_arr, axis=1),
                     np.std(total_acc_x_arr, axis=1), np.std(total_acc_y_arr, axis=1),
                     np.std(total_acc_z_arr, axis=1))).T
    # 9,var方差
    var = np.vstack((np.var(gyro_x_arr, axis=1), np.var(gyro_y_arr, axis=1), np.var(gyro_z_arr, axis=1),
                     np.var(body_x_arr, axis=1), np.var(body_y_arr, axis=1), np.var(body_z_arr, axis=1),
                     np.var(total_acc_x_arr, axis=1), np.var(total_acc_y_arr, axis=1),
                     np.var(total_acc_z_arr, axis=1))).T
    # 10, energy能量
    energy = np.vstack(
        (np.sum(np.abs(gyro_x_arr), axis=1), np.sum(np.abs(gyro_y_arr), axis=1), np.sum(np.abs(gyro_z_arr), axis=1),
         np.sum(np.abs(body_x_arr), axis=1), np.sum(np.abs(body_y_arr), axis=1), np.sum(np.abs(body_z_arr), axis=1),
         np.sum(np.abs(total_acc_x_arr), axis=1), np.sum(np.abs(total_acc_y_arr), axis=1),
         np.sum(np.abs(total_acc_z_arr), axis=1))).T
    # 11,entropy信息熵
    entropy_gyro_x = ((-(gyro_x_arr / gyro_x_arr.sum()) * np.log2(gyro_x_arr / gyro_x_arr.sum())).sum(axis=1))
    entropy_gyro_y = ((-(gyro_y_arr / gyro_y_arr.sum()) * np.log2(gyro_y_arr / gyro_y_arr.sum())).sum(axis=1))
    entropy_gyro_z = ((-(gyro_z_arr / gyro_z_arr.sum()) * np.log2(gyro_z_arr / gyro_z_arr.sum())).sum(axis=1))
    entropy_body_x = ((-(body_x_arr / body_x_arr.sum()) * np.log2(body_x_arr / body_x_arr.sum())).sum(axis=1))
    entropy_body_y = ((-(body_y_arr / body_y_arr.sum()) * np.log2(body_y_arr / body_y_arr.sum())).sum(axis=1))
    entropy_body_z = ((-(body_z_arr / body_z_arr.sum()) * np.log2(body_z_arr / body_z_arr.sum())).sum(axis=1))
    entropy_gra_x = (
        (-(total_acc_x_arr / total_acc_x_arr.sum()) * np.log2(total_acc_x_arr / total_acc_x_arr.sum())).sum(axis=1))
    entropy_gra_y = (
        (-(total_acc_y_arr / total_acc_y_arr.sum()) * np.log2(total_acc_y_arr / total_acc_y_arr.sum())).sum(axis=1))
    entropy_gra_z = (
        (-(total_acc_z_arr / total_acc_z_arr.sum()) * np.log2(total_acc_z_arr / total_acc_z_arr.sum())).sum(axis=1))
    entropy = np.vstack((entropy_gyro_x, entropy_gyro_y, entropy_gyro_z,
                         entropy_body_x, entropy_body_y, entropy_body_z,
                         entropy_gra_x, entropy_gra_y, entropy_gra_z)).T
    # 12,cov协方差
    cov_gyro_xy = []
    cov_gyro_xz = []
    cov_gyro_yz = []
    cov_gra_xy = []
    cov_gra_xz = []
    cov_gra_yz = []
    cov_body_xy = []
    cov_body_xz = []
    cov_body_yz = []
    for i in range(0, len(gyro_x_arr)):
        cov_gyro_xy.append(np.cov(gyro_x_arr[i], gyro_y_arr[i])[0][1])
        cov_gyro_xz.append(np.cov(gyro_x_arr[i], gyro_z_arr[i])[0][1])
        cov_gyro_yz.append(np.cov(gyro_y_arr[i], gyro_z_arr[i])[0][1])
        cov_gra_xy.append(np.cov(total_acc_x_arr[i], total_acc_y_arr[i])[0][1])
        cov_gra_xz.append(np.cov(total_acc_x_arr[i], total_acc_z_arr[i])[0][1])
        cov_gra_yz.append(np.cov(total_acc_y_arr[i], total_acc_z_arr[i])[0][1])
        cov_body_xy.append(np.cov(body_x_arr[i], body_y_arr[i])[0][1])
        cov_body_xz.append(np.cov(body_x_arr[i], body_z_arr[i])[0][1])
        cov_body_yz.append(np.cov(body_y_arr[i], body_z_arr[i])[0][1])
    cov = np.vstack((np.array(cov_gyro_xy), np.array(cov_gyro_xz), np.array(cov_gyro_yz),
                     np.array(cov_gra_xy), np.array(cov_gra_xz), np.array(cov_gra_yz),
                     np.array(cov_body_xy), np.array(cov_body_xz), np.array(cov_body_yz))).T
    
    # 特征拼接
    all_features = np.hstack((corr_gust, mad, IQR, mean, median, max, min, std, var, energy, cov))
    print("All features's shape：", all_features.shape)
    return all_features


# 执行方法
# For train set
all_features_UCI = features_extraction(gyro_x_arr, gyro_y_arr, gyro_z_arr, body_x_arr, body_y_arr, body_z_arr,
                                       total_acc_x_arr, total_acc_y_arr, total_acc_z_arr)
# For test set (F)
all_features_test = features_extraction(test_gyro_x_arr, test_gyro_y_arr, test_gyro_z_arr, test_body_x_arr,
                                        test_body_y_arr, test_body_z_arr, test_total_acc_x_arr, test_total_acc_y_arr,
                                        test_total_acc_z_arr)
# For test set (UCI)
all_features_test_UCI = features_extraction(UCI_test_gyro_x_arr, UCI_test_gyro_y_arr, UCI_test_gyro_z_arr,
                                            UCI_test_body_x_arr,
                                            UCI_test_body_y_arr, UCI_test_body_z_arr, UCI_test_total_acc_x_arr,
                                            UCI_test_total_acc_y_arr,
                                            UCI_test_total_acc_z_arr)
# Write train features
all_features_UCI_df = pd.DataFrame(all_features_UCI)
all_features_UCI_df.to_csv('Features_UCI.csv', index_label=False)
all_features_UCI_df.describe().to_csv('Features_UCI_describe.csv')
# Write test features(F)
all_features_test_df = pd.DataFrame(all_features_test)
all_features_test_df.to_csv('Features_final.csv', index_label=False)
all_features_test_df.describe().to_csv('Features_final_describe.csv')
# Write test features(UCI)
all_features_test_UCI_df = pd.DataFrame(all_features_test_UCI)
all_features_test_UCI_df.to_csv('Features_test_UCI.csv', index_label=False)
all_features_test_UCI_df.describe().to_csv('Features_test_UCI_describe.csv')
# All UCI data set features
features_all_UCI = np.vstack((all_features_UCI, all_features_test_UCI))
features_all_UCI_df = pd.DataFrame(features_all_UCI)
features_all_UCI_df.to_csv('Features_all_UCI.csv', index_label=False)
features_all_UCI_df.describe().to_csv('Features_all_UCI_describe.csv', index_label=False)
