import scipy.io as sio
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from torch.autograd import Variable
import math
import csv
import Data_Input
from sklearn.preprocessing import StandardScaler

import Read_test_signal_csv
import read_features


# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(hidden_size, output_size)  # 全连接层
    
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x


if __name__ == '__main__':
    
    # checking if GPU is available
    device = torch.device("cpu")
    
    if (torch.cuda.is_available()):
        device = torch.device("cuda:0")
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')
    
    # 读取数据，nd.array类型
    # train_x = Data_Input.return_X_train_Array().astype('float32')
    # train_y = Data_Input.return_y_train_Array()
    #
    # test_x = Data_Input.return_X_test_Array().astype('float32')
    # test_y = Data_Input.return_y_test_Array()

    train_x, test_x, train_y, test_y = train_test_split(read_features.return_all_features_final(),
                                                    Read_test_signal_csv.return_final_labesl(), test_size=0.20, train_size=0.80,
                                                    random_state=43)

train_x_len = len(train_x)
t_train = np.linspace(0, train_x_len, train_x_len)

test_x_len = len(test_x)
t_test = np.linspace(0, test_x_len, test_x_len)

"""train_data_ratio = 0.8  # Choose 80% of the data for training
train_data_len = int(data_len * train_data_ratio)

train_x = data_x[5:train_data_len]
train_y = data_y[5:train_data_len]
t_for_training = t[5:train_data_len]

test_x = data_x[train_data_len:]
test_y = data_y[train_data_len:]
t_for_testing = t[train_data_len:]"""

# ----------------- train -------------------
INPUT_FEATURES_NUM = 561
OUTPUT_FEATURES_NUM = 1
train_x_tensor = train_x.reshape(-1, 1, INPUT_FEATURES_NUM)  # set batch size to 1
train_y_tensor = train_y.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batch size to 1

# transfer data to pytorch tensor
train_x_tensor = torch.from_numpy(train_x_tensor)
train_y_tensor = torch.from_numpy(train_y_tensor)

lstm_model = LstmRNN(INPUT_FEATURES_NUM, 20, output_size=OUTPUT_FEATURES_NUM, num_layers=1)  # 20 hidden units
print('LSTM model:', lstm_model)
print('model.parameters:', lstm_model.parameters)
print('train x tensor dimension:', Variable(train_x_tensor).size())

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

# epoch
prev_loss = 1000
max_epochs = 1000

train_x_tensor = train_x_tensor.to(device)
train_losses = []
for epoch in range(max_epochs):
    output = lstm_model(train_x_tensor).to(device)
    loss = criterion(output.float(), train_y_tensor.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if loss < prev_loss:
        torch.save(lstm_model.state_dict(), 'lstm_model.pt')  # save model parameters to files
        prev_loss = loss
    
    if loss.item() < 1e-4:
        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
        print("The loss value is reached")
        break
    elif (epoch + 1) % 100 == 0:
        print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))
    train_losses.append(loss.item())
# prediction on training dataset
pred_y_for_train = lstm_model(train_x_tensor).to(device)
pred_y_for_train = pred_y_for_train.view(-1, OUTPUT_FEATURES_NUM).data.numpy()
# ----------------- test -------------------
lstm_model = lstm_model.eval()  # switch to testing model

# prediction on test dataset
test_x_tensor = test_x.reshape(-1, 1,
                               INPUT_FEATURES_NUM)
test_x_tensor = torch.from_numpy(test_x_tensor)  # 变为tensor
test_x_tensor = test_x_tensor.to(device)

pred_y_for_test = lstm_model(test_x_tensor).to(device)
pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.numpy()
print(pred_y_for_test)
loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(test_y))
print("test loss：", loss.item())

# 画图部分，暂时不用
# ----------------- plot -------------------
plt.figure()
# plt.subplot(1,2,1)
# plt.plot(np.array(range(max_epochs)), np.array(train_accuracies))
# plt.title('SSE--Cluster Intertia')
# plt.xlabel('Cluster')
# plt.ylabel('SSE')
# plt.subplot(1,2,2)
plt.plot(np.array(range(max_epochs)), np.array(train_losses))
# plt.title('Davies Bouldin Score')
plt.xlabel('Epoches')
plt.ylabel('Losses')
plt.show()
