# @author : Yulin Huang
# @email : yulinhuang0217@gmail.com
# @software: pycharm
import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import Data_Input
from sklearn import svm, metrics
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
import Data_Input
import warnings

import Read_test_signal_csv
import read_features

# 忽略警告
warnings.filterwarnings("ignore")


# x_train = read_features.return_features_whole_UCI_10299_items()
# y_train = Data_Input.return_all_labels()
# x_test = read_features.return_all_features_test_UCI()
# y_test = Data_Input.return_y_test_Array()
x_train, x_test, y_train, y_test = train_test_split(read_features.return_all_features_final(),
                                                    Read_test_signal_csv.return_final_labesl(), test_size=0.20, train_size=0.80,
                                                    random_state=43)
# Training
model = GaussianNB()
print("[INFO] Successfully initialize a new model !")
print("[INFO] Training the model…… ")
model.fit(x_train, y_train)
print("[INFO] Model training completed !")
y_test_pred = model.predict(x_test)
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
print("[INFO] Successfully get Naive_Bayes_Gaussian's classification overall accuracy ! ")


def get_y():
    return y_test, y_test_pred
