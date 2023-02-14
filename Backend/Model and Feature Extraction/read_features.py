# @author : Yulin Huang
# @email : yulinhuang0217@gmail.com
# @software: pycharm
import numpy as np
import pandas as pd


def read_file(Path):
    """Read the .txt, .data or .csv file as dataframe. Input a parameter 'Path', which indicates the location of the file."""
    dataframe = pd.read_csv(filepath_or_buffer=Path, header=0)
    return dataframe


def get_array(dataframe):
    X = np.array(dataframe)
    return X


# Read train features
all_features_UCI_df = read_file('Features_UCI.csv')
all_features_UCI = get_array(all_features_UCI_df)
# Read test features(F)
all_features_final_df = read_file('Features_final.csv')
all_features_final = get_array(all_features_final_df)
# Read test features(UCI)
all_features_test_UCI_df = read_file('Features_test_UCI.csv')
all_features_test_UCI = get_array(all_features_test_UCI_df)
# Read all UCI features
features_all_UCI_df = read_file('Features_all_UCI.csv')
features_all_UCI = get_array(features_all_UCI_df)


def return_all_features_UCI():
    """Return the merged UCI features"""
    return all_features_UCI


def return_all_features_final():
    """Return the merged test features(F)"""
    return all_features_final


def return_all_features_test_UCI():
    """Return the merged test features(UCI)"""
    return all_features_test_UCI


def return_features_whole_UCI_10299_items():
    """Return the merged all features(UCI) including train set and test set"""
    return features_all_UCI
