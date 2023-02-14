# @author : Yulin Huang
# @email : yulinhuang0217@gmail.com
# @software: pycharm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from pprint import pprint

import warnings

import Data_Input

warnings.filterwarnings("ignore")

X_train = Data_Input.return_X_train_DF()
X_test = Data_Input.return_X_test_DF()
y_train = Data_Input.return_y_train_DF()
y_test = Data_Input.return_y_test_DF()

# 构建神经网络模型
# 隐藏层数量及其神经元数量
hidden_layer_sizes = (10, 20, 30)
mlpclassifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
mlpclassifiermodel = mlpclassifier.fit(X_train, y_train)
print('神经网络模型', '-' * 30, '\n', mlpclassifiermodel)
print('神经网络模型得分', '-' * 30, '\n', mlpclassifiermodel.score(X_test, y_test))
print('神经网络模型系数,前1个', '-' * 30, '\n', mlpclassifiermodel.coefs_[0])
print('神经网络模型截距', '-' * 30, '\n', mlpclassifiermodel.intercepts_)
print('神经网络训练误差', '-' * 30, '\n', mlpclassifiermodel.loss_)
print("神经网络模型参数")
pprint(mlpclassifiermodel.get_params())

# 测试模型
y_predict = mlpclassifiermodel.predict(X_test)
# 打印真实值与预测值
y_predict_df = pd.DataFrame(y_predict, columns=['y_predict'], index=y_test.index)
y_test_predict_df = pd.concat([y_test, y_predict_df], axis=1)
print('神经网络真实值与预测值', '-' * 30, '\n', y_test_predict_df)
# 模型优度的可视化展现
fpr, tpr, _ = metrics.roc_curve(y_test, y_predict, pos_label=2)
auc = metrics.auc(fpr, tpr)

plt.style.use('ggplot')  # 设置绘图风格
plt.plot(fpr, tpr, '')  # 绘制ROC曲线
plt.plot((0, 1), (0, 1), 'r--')  # 绘制参考线
plt.text(0.5, 0.5, 'AUC=%.2f' % auc)  # 添加文本注释
plt.title('MLPClassifier ROC')  # 设置标题
plt.xlabel('False Positive Rate')  # 设置坐标轴标签
plt.ylabel('True Positive Rate')
plt.tick_params(top='off', right='off')  # 去除图形顶部边界和右边界的刻度
plt.show()  # 图形显示

##########################################################
# 模型评估
# F1 = 2 * (precision * recall) / (precision + recall)
accuracy = metrics.accuracy_score(y_test, y_predict)
confusionmatrix = metrics.confusion_matrix(y_test, y_predict)
target_names = ['label 0', 'label 1', 'label 2', 'label 3', 'label 4', 'label 5']
classifyreport = metrics.classification_report(y_test, y_predict, target_names=target_names)
print('神经网络分类准确率 ', accuracy)  # 混淆矩阵对角线元素之和/所有元素之和
print('神经网络混淆矩阵 \n', confusionmatrix)
print('神经网络分类结果报告 \n', classifyreport)

# 优化模型,选择最佳参数
parameters = {
    'activation': ('logistic', 'relu'),
    'solver': ('lbfgs', 'sgd', 'adam'),
    'learning_rate': ('constant', 'invscaling', 'adaptive'),
}

grid_search = GridSearchCV(MLPClassifier(), parameters, verbose=0, scoring='accuracy', cv=5)
grid = grid_search.fit(X_train, y_train)
print('神经网络最佳效果：%0.3f' % grid_search.best_score_)

best_parameters = grid_search.best_estimator_.get_params()
print('神经网络最佳参数')
pprint(best_parameters)