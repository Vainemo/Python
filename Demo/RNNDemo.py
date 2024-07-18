from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math 
from sklearn.metrics import mean_squared_error
window_size=3
#制作正弦波数据
def sum_of_sines(number_of_steps, d_theta, skip_steps,
 freqs, amps, phases):
 values = []
 for step_num in range(number_of_steps):
  angle = d_theta * (step_num + skip_steps)
  sum = 0
  for wave in range(len(freqs)):
    y = amps[wave] * math.sin(freqs[wave]*(phases[wave] + angle))
    sum += y
  values.append(sum)
 return np.array(values)

#创建数据
train_sequence= sum_of_sines(200, 0.057, 0,
 [1.1, 1.7, 3.1, 7],
 [1, 2, 2, 3], [0, 0, 0, 0])
test_sequence = sum_of_sines(200, 0.057, 400,
 [1.1, 1.7, 3.1, 7],
 [1, 2, 2, 3], [0, 0, 0, 0])

#将列表数据（x，y）转化为一列元素(按照X数据排列的一列数据)归一化操作需要
#将数据格式为按特征垂直排列
train_sequence = np.reshape(train_sequence,(train_sequence.shape[0], 1))
test_sequence = np.reshape(test_sequence,(test_sequence.shape[0], 1))
#将数据进行归一化并且训练
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
min_max_scaler.fit(train_sequence)
#转换应用于测试和验证数据上，我们之后能够将它的逆转换应用于神经网络的输出，来给出一个与输入相同范围的结果
scaled_train_sequence = min_max_scaler.transform(train_sequence)
scaled_test_sequence = min_max_scaler.transform(test_sequence)

#将一个列表的元素及窗口尺寸转换为两个新列表
#第一个新列表包含多个由初始列表产生的相互重叠的子序列。每个子序列长度比给
#定的窗口尺寸小1。第二个新列表则包含初始序列中每个上述子序列的下一个元素，
#它在训练和测试中将被用作我们的目标。
def samples_and_targets_from_sequence(sequence, window_size):
 samples = []
 targets = []
 for i in range(sequence.shape[0]-window_size):
  sample = sequence[i:i+window_size]
  target = sequence[i+window_size]
  samples.append(sample)
  targets.append(target[0])
  return (np.array(samples), np.array(targets))
#窗口化的训练数据
(X_train, y_train) = samples_and_targets_from_sequence(
 scaled_train_sequence, window_size)
#窗口化的测试数据
(X_test, y_test) = samples_and_targets_from_sequence(
 scaled_test_sequence, window_size)
#创建一个LSTM层对象。第一个参数是这层中LSTM单元的数量。第二个参数是input_shape，告
#诉了单个样本的尺寸。

lstm_layer = LSTM(3, input_shape=[window_size, 1]
model = Sequential()
model.add(lstm_layer)
model.add(Dense(1, activation=None))
#选择mean_squared_error损失函数:将网络输出的单一值与该样本的目标值进行比较的函数
model.compile(loss='mean_squared_error', optimizer='adam')
#批 量大小为1
history = model.fit(X_train, y_train, epochs=number_of_epochs,batch_size=1, verbose=2)
#训练数据和测试数据两者的预测
y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)
#方向逆转预测数据（之前输入的数据是经过归一化处理的，现在需要转换）
#[]:将元素作为一个列表输入y_train为该列表的第一个元素
inverse_y_train = min_max_scaler.inverse_transform([y_train])
inverse_y_test = min_max_scaler.inverse_transform([y_test])
inverse_y_train_predict =min_max_scaler.inverse_transform(y_train_predict)
inverse_y_test_predict =min_max_scaler.inverse_transform(y_test_predict)
#计算并报告训练和测试集的预测结果的RMS误差。
#[：，0]获得第一列的数据
trainScore = math.sqrt(mean_squared_error(inverse_trainY[0],inverse_y_train_predict[:,0]))
print('Training RMS error: {:.2f}'.format(trainScore))
testScore = math.sqrt(mean_squared_error(inverse_testY[0],inverse_y_test_predict[:,0]))
print('Test RMS error: {:.2f}'.format(testScore))


#创建一个更加复杂的正弦数据集合
def sum_of_upsloping_sines(number_of_steps, d_theta,skip_steps, freqs, amps, phases):
 values = []
 #range 函数在 Python 中用于生成一个整数序列0-number_of_steps
 for step_num in range(number_of_steps):
   angle = d_theta * (step_num + skip_steps)
   sum = 0
   for wave in range(len(freqs)):
      y = amps[wave] * math.sin(freqs[wave]*(phases[wave] + angle))
      sum += y
   values.append(sum)
 if step_num > 0: # are we past the first sample?
 # find the direction we′re headed in
    sum_change = sum - prev_sum
    if sum_change < 0: # are we going downward?
      values[-1] *= -1 # if so, flip the last

# 组建一个深度RNN仅仅意味着要添加更多的循环层。所有在另一个循环层之前的循环层都必须将它们的
# 可选参数return_sequences设置为True。原因：假设输入数据是时间步为5，特征数量为1的数据，想预测
#时间步6的数据，如果是一层循环层，我们只关系他对第六步的预测，如果预测值需要输入到下一层循环层
#只有一个数据的话那就丢失了时间步信息，所有需要将return_sequences设置为true，将每一步的信息都输出出来
model = Sequential()
model.add(LSTM(3, return_sequences=True,nput_shape=[window_size, 1]))
model.add(LSTM(3))
model.add(Dense(1))
#创建一个有状态的RNN，我们需要给第一个LSTM一个batch_size的值（训练期间使用的批处理大小。），
# 并在每个LSTM中设定stateful=True（表示手动控制什么时候清除状态）
model = Sequential()
model.add(LSTM(50,input_shape=(time_steps, 1),return_sequences=True,batch_size=batch_size, stateful=True))
model.add(LSTM(50, stateful=True))
model.add(Dense(1))
#训练一个有状态的RNN。我们需要告诉fit()不要打乱数据，shuffle=False,然后需要在每个epoch之后调用reset_ states()
for i in range(number_of_epochs):
 model.fit(X_train, y_train, batch_size=batch_size,
 epochs=1, verbose=1, shuffle=False)
 model.reset_states()
#使用TimeDistributed包装全连接层，使全连接层按照时间步依次处理数据，我们可以使用一个TimeDistributed层包装多个层，也可以单独包装每一个。
 model = Sequential()
model.add(LSTM(4, return_sequences=True,
 input_shape=[window_size, 1]))
model.add(TimeDistributed(Dense(5))