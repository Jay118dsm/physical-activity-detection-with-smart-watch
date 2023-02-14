from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut, ShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import svm, metrics
from sklearn.metrics import classification_report
import Data_Input
import warnings

import Read_test_signal_csv
import read_features

# 忽略警告
warnings.filterwarnings("ignore")

# x_train = Data_Input.return_X_train_Array()
# x_test = Data_Input.return_X_test_Array()
# y_train = Data_Input.return_y_train_Array()
# y_test = Data_Input.return_y_test_Array()
x_train, x_test, y_train, y_test = train_test_split(read_features.return_all_features_final(),
                                                    Read_test_signal_csv.return_final_labesl(), test_size=0.20, train_size=0.80,
                                                    random_state=43)
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 训练模型
rf = RandomForestClassifier()
param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
print("[INFO] Successfully initialize a new model !")
print("[INFO] Training the model…… ")
# 网格搜索与交叉验证
gc = GridSearchCV(rf, param_grid=param, cv=2)  # cv表示验证的次数为2
gc.fit(x_train, y_train)
print("Hyper-parameter Selection：", gc.best_params_)
print("[INFO] Model training completed !")
# 测试模型
global y_test_pred
y_test_pred = gc.predict(x_test)
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



def get_y():
    return y_test, y_test_pred
