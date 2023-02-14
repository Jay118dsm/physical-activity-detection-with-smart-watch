# @author : Yulin Huang
# @email : yulinhuang0217@gmail.com
# @software: pycharm
import numpy as np
import pandas as pd

def read_test_file(Path):
    """Read the .txt, .data or .csv file as dataframe. Input a parameter 'Path', which indicates the location of the file."""
    dataframe = pd.read_csv(filepath_or_buffer=Path, header=0)
    return dataframe

def get_array(dataframe):
    X = np.array(dataframe)
    return X

#全部按gra_xAcc长度删减其他文件长度为213行（手动删减）
body_x = read_test_file('finish_data/Upstairs(2)/body_xAcc_final.csv')
body_x_arr = get_array(body_x)
body_y = read_test_file('finish_data/Upstairs(2)/body_yAcc_final.csv')
body_y_arr = get_array(body_y)
body_z = read_test_file('finish_data/Upstairs(2)/body_zAcc_final.csv')
body_z_arr = get_array(body_z)
#Merge
body = np.hstack((body_x_arr,body_y_arr,body_z_arr))
#print(body)

"""#全部按gra_xAcc长度删减其他文件长度为213行（手动删减）
gra_x = read_file('Data F/gra_xAcc_processed.csv')
gra_x_arr = get_array(gra_x)
gra_y = read_file('Data F/gra_yAcc_processed.csv')
gra_y_arr = get_array(gra_y)
gra_z = read_file('Data F/gra_zAcc_processed.csv')
gra_z_arr = get_array(gra_z)
#Merge
gra =  np.hstack((gra_x_arr,gra_y_arr,gra_z_arr))"""


#全部按gra_xAcc长度删减其他文件长度为213行（手动删减）
gry_x = read_test_file('finish_data/Upstairs(2)/xGyr_final.csv')
gry_x_arr = get_array(gry_x)
gry_y = read_test_file('finish_data/Upstairs(2)/yGyr_final.csv')
gry_y_arr = get_array(gry_y)
gry_z = read_test_file('finish_data/Upstairs(2)/zGyr_final.csv')
gry_z_arr = get_array(gry_z)
#Merge
gry = np.hstack((gry_x_arr,gry_y_arr,gry_z_arr))
#print(gyro)

final = np.hstack((body,gry))


labels = read_test_file('finish_data/labels_final.csv').astype(int)
labels_arr = get_array(labels)


def return_test_signal():
    """Return final merged ndarray"""
    return final

def return_final_labesl():

    return labels_arr



