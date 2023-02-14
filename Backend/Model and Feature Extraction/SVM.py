# @author : Yulin Huang
# @email : yulinhuang0217@gmail.com
# @software: pycharm
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm, metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
import Data_Input
import warnings

import PCA
import Read_test_signal_csv
import read_features
import joblib
import pandas as pd

# 忽略警告
warnings.filterwarnings("ignore")

y_test = []
y_test_pred = []


def train_and_test_model(feature_train, label_train, feature_test, label_test, confusion_matrix=None):
    # label_train = LabelBinarizer().fit_transform(label_train)
    # label_test = LabelBinarizer().fit_transform(label_test)
    # 随机数种子
    random_state = np.random.RandomState(0)
    # 划分训练测试数据
    X_train = feature_train
    y_train = label_train
    X_test = feature_test
    global y_test
    y_test = label_test
    # 训练模型
    model = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
    print("[INFO] Successfully initialize a new model !")
    print("[INFO] Training the model…… ")
    clt = model.fit(X_train, y_train)
    # clt = joblib.load('models/SVM_siganl.pkl')
    print("[INFO] Model training completed !")
    # 测试模型
    global y_test_pred
    y_test_pred = clt.predict(X_test)
    ov_acc = metrics.accuracy_score(y_test_pred, y_test)
    print("Overall accuracy: %f" % (ov_acc))
    print("===========================================")
    acc_for_each_class = metrics.precision_score(y_test, y_test_pred, average=None)
    print("Accuracy for each class:\n", acc_for_each_class)
    print("===========================================")
    avg_acc = np.mean(acc_for_each_class)
    print("Average accuracy(For all classes):%f" % (avg_acc))
    print("===========================================")
    classification_rep = classification_report(y_test, y_test_pred)
    print("Classification report: \n", classification_rep)
    print("===========================================")
    # print("===========================================")
    confusion_matrix = metrics.confusion_matrix(y_test, y_test_pred)
    print("Confusion metrix:\n", confusion_matrix)
    print("===========================================")
    # print("accuracy: %f"%(acc_r))
    print("[INFO] Successfully get SVM's classification overall accuracy ! ")
    return model


# 加载特征和标签文件 For feature Extracted data
"""feature_train = Data_Input.return_X_train_Array()
label_train = Data_Input.return_y_train_Array()
feature_test = Data_Input.return_X_test_Array()
label_test = Data_Input.return_y_test_Array()"""

# 加载特征和标签文件 For Inertial Signal data(self extracted)
feature_train = PCA.return_PCA3()
label_train = Data_Input.return_all_labels()
feature_test = read_features.return_all_features_final()
label_test = Read_test_signal_csv.return_final_labesl()

feature_train, feature_test, label_train, label_test = train_test_split(read_features.return_all_features_final(),
                                                                        Read_test_signal_csv.return_final_labesl(), test_size=0.20, train_size=0.80,
                                                                        random_state=43)

# 训练与测试模型

model = train_and_test_model(feature_train, label_train, feature_test, label_test)
# joblib.dump(model, 'models/SVM_siganl(特征提取惯性数据(All UCI)).pkl')
print(y_test_pred)

# Load the Model
# new_model = joblib.load('models/SVM_siganl(特征提取惯性数据(All UCI)).pkl')
# y_test_pred = new_model.predict(feature_test)
# print(y_test_pred)
# ov_acc = metrics.accuracy_score(y_test_pred, label_test)
# print("Overall accuracy: %f" % (ov_acc))
# print("===========================================")
# acc_for_each_class = metrics.precision_score(label_test, y_test_pred, average=None)
# print("Accuracy for each class:\n", acc_for_each_class)
# print("===========================================")
# avg_acc = np.mean(acc_for_each_class)
# print("Average accuracy(For all classes):%f" % (avg_acc))
# print("===========================================")
# classification_rep = classification_report(label_test, y_test_pred)
# print("Classification report: \n", classification_rep)
# print("===========================================")
# # print("===========================================")
# confusion_matrix = metrics.confusion_matrix(label_test, y_test_pred)
# print("Confusion metrix:\n", confusion_matrix)
# print("===========================================")
# # print("accuracy: %f"%(acc_r))
# print("[INFO] Successfully get SVM's classification overall accuracy ! ")
