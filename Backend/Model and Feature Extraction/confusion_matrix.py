# @author : Yulin Huang
# @email : yulinhuang0217@gmail.com
# @software: pycharm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

import GaussianNB
# SVM模型混淆矩阵
# import SVM
# y_true, y_pred = SVM.get_y()

# Bayes模型混淆矩阵
# import GaussianNB
# y_true, y_pred = GaussianNB.get_y()

# Random_Forest模型混淆矩阵
import Random_Forest

y_true, y_pred = GaussianNB.get_y()

# labels表示不同类别的代号
labels = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Cycling', 'Jumping','Running']
tick_marks = np.array(range(len(labels))) + 0.5


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix(GaussianNB)')
plt.show()
