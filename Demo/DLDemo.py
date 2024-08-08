import tensorflow as tf
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import backend as keras_backend
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
random_seed = 42
#import yaml
import json
np.random.seed(random_seed)
# 加载MNIST数据集
mnist = tf.keras.datasets.mnist
# 例程mnist.load_data()返回一个训练集和一个测试集
# 用大写字母X来表示数据集的样本，使用小写字母y来表示其标签
#(离开,向下,向右)约定
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#保存输入数据的图像大小
image_height = X_train.shape[1]
image_width = X_train.shape[2]
number_of_pixels = image_height * image_width
#将数组内容类型转化为浮点数
X_train = keras_backend.cast_to_floatx(X_train)
X_test = keras_backend.cast_to_floatx(X_test)

#把所有的图像做归一化处理（每张图片都是以每个像素的灰度值组成的数组来表示（0-255之间））
x_train, x_test = X_train / 255.0, X_train / 255.0

#--------------------------------------------------------------------------------------------------------
#将所有标签组合成一个大列表并提取其最大值。由于我们从0开始，因此将为结果添加1，这是可以编码所有标签中所有值的最小列表大小。
number_of_classes = 1 + max(np.append(y_train, y_test))
#将整数标签列表转换为独热编码列表
y_train = to_categorical(y_train, num_classes=number_of_classes)
y_test = to_categorical(y_test, num_classes=number_of_classes)
#--------------------------------------------------------------------
#保存原始数据（数据和标签）
original_y_train = y_train
original_y_test = y_test
#--------------------------------------------------------------------
#将独热编码表示为常规Python列表（即不是NumPy数组）,index方法
one_hot = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
one_hot.index(1)
#表示为NumPy数组：argmax方法
one_hot_np = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
np.argmax(one_hot_np)
#-------------------------------------------------------------------------------------------------------
# 在这段代码中，我们反复覆盖X_train和X_test中的数据，以及y_train和y_test中的标签。这是预处理过程中的常用方法，
# 因为我们不关心起始值或中间值。这种方法的好处是它带来了一定程度的简单性；缺点是如果我们想要访问原始数据，要么必
# 须保存它（就像我们在这里为标签所做的那样），要么加载数据的新副本 
#------------------------------------------------------------------------------------------------------- 

#将图像展平为二维网格，因此每个样本只是一个数字列表（代表一张图片的信息）。这是全连接层所需的格式
#X_train = np.reshape(X_train,[X_train.shape[0], number_of_pixels])
#X_test = np.reshape(X_test, [X_test.shape[0], number_of_pixels])
#第二种方法
X_train = X_train.reshape([X_train.shape[0], number_of_pixels])
X_test = X_test.reshape([X_test.shape[0], number_of_pixels])

#Sequential：层列表架构 Functional：自定义列表架构
#激活函数的选择：
# 常见的选择是将“relu”和“tanh”用于隐藏层，将“softmax”或“sigmoid”用于输出层。默认值为“none”或线性激活函数。
#-----------------------------------------------------------------
#建立层列表架构
def  make_one_hidden_layer_model():
  model = tf.keras.models.Sequential()
#创建一个全连接层，第一个参数：参数是层的大小，第二个参数：激活函数的类型，第三个参数：输入中每个维度的大小，input_shape只针对第一层
  model.add(tf.keras.layers.Dense(number_of_pixels, activation='relu',input_shape=[number_of_pixels]))
#再次添加一个全连接层 number_of_classes（第一个参数）：神经元的数量，第二个参数：激活函数的类型
  model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))
#输出文本形式的模型
  model.summary()
#自定义一个优化器，学习率设定为0.0001
  slow_adam = tf.keras.optimizers.Adam(lr=0.0001)
#编译模型，指定优化器为slow_adam，损失函数为categorical_crossentropy，测量值列表返回accuracy来记录准确率
  model.compile(optimizer=slow_adam,loss='categorical_crossentropy',metrics=['accuracy'])
  return model
#调用方法
model = make_one_hidden_layer_model() 
#---------------------------------------------------------------------------------------------------
#1.检查点
#在训练期间检查我们的模型。这意味着将模型（或者，如果我们愿意，只是权重）保存到文件中 save_weights_only:只保存权重
#period：10个epochs记录一次
filename = 'SavedModels/weights-{epoch:02d}-{val_loss:.03f}.h5'
filename += 'epoch-{epoch:03d}-acc-{acc:0.3f}.h5'
#请注意，由于我们只输出了值的3位数，因此准确率可能没有明显提高。例如，如果它从0.9353变为0.9354，则两个文件都会将文件
# 名中的准确率列为0.935。通过查看文件的时间戳，我们可以推断出最近生成的文件更好
#tf.keras.callbacks.ModelCheckpoint 回调允许您在训练期间和结束时持续保存模型。
checkpointer = tf.keras.ModelCheckpoint(filename, monitor='acc',save_weights_only=True,period=10)
#---------------------------------------------------------------------------------------------------
#2.学习率
sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
#将初始的学习率设置为0;
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
#同光方法不断调整学习率
def simpleSchedule(epoch_number):
 return max(.1, 1-(0.01*epoch_number))
lr_scheduler = LearningRateScheduler(simpleSchedule)
one_hidden_layer_history =model.fit(x_train, y_train,validation_data=(X_test, y_test), epochs=20,batch_size=256, verbose=2,callbacks = [lr_scheduler])

#3.及早停止
#monitor:可以指定关注哪个参数：训练准确率“acc”、训练损失“loss”、验证准确率“val_acc”或验证损失“val_loss”
#min_delta：min_delta是变化的监测值EarlyStopping()开始起作用的最小值。任何小于此数量的改变都将被忽略。默认情况下，此值为0，
#patience：这是在决定fit( )应该停止训练之前等待情况好转的epoch数
#verbose：如果它决定停止训练，则输出一行文本，这样我们就可以查看输出并知道它进行了干预。
early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
history = model.fit(X_train, y_train,validation_data=(X_test, y_test),epochs=100, batch_size=256, verbose=2,callbacks=[early_stopper])

#1.训练模型
#2保存历史记录，它包含一堆总结训练过程的字段（比如它运行了多少个epoch，以及我们使用了哪些参数）。
#one_hidden_layer_history：它是一个Python字典对象，包含每一个epoch后训练集和验证集的准确率和损失值。
#batch_size:告诉fit( )从我们的训练集中提取出一个批大小的样本块应该有多大
#verbose:它告诉系统在每一个epoch之后的更新结果,如果我们将其设置为0，则不输出任何内容；值为1会输出一个动态进度条，
# 显示系统在每一个epoch通过样本的方式；值为2则仅输出每一个epoch后的单个文本摘要
one_hidden_layer_history =model.fit(x_train, y_train,validation_data=(X_test, y_test), epochs=20,batch_size=256, verbose=2,callbacks = [checkpointer])

#训练准确率
one_hidden_layer_history['acc']
#训练损失率
one_hidden_layer_history['loss']
#验证准确率
one_hidden_layer_history['val_acc']
#验证损失率
one_hidden_layer_history['val_loss']
#保存模型有三种结构：
  #1.keras（推荐使用的）
  #2.HDF5
  #3.SavedModel 
#区别：.keras/HDF5 格式使用对象配置来保存模型架构，而 SavedModel 保存执行计算图
#因此，SavedModels 能够保存自定义对象，例如子类化模型和自定义层，而无需原始代码
#保存模型和权重
model.save('my_model.h5')
#加载模型
model = tf.keras.load_model('my_model.h5')
#仅保存权重 如果我们想以后使用这些权重，那么必须首先创建一个模型来接收它们
model.save_weights('my_model_weights.h5')
#仅保存框架:
# 1.保存为yaml形式，yaml是JSON的超集，它可以完成JSON能做的所有事情
filename = 'my_model_arch.yaml'
yaml_string = model.to_yaml()
with open(filename, 'w') as outfile:
#yaml.dump(yaml_string, outfile)
#2.保存为json格式，只需要替换所有yaml
#filename = 'my_model_arch_json.json'
#json_string = model.to_json()
#with open(filename, 'w') as outfile:
#json.dump(json_string, outfile)
#读取框架
#model =tf.keras.models.model_from_yaml(yaml_string)
#无论是分享我们自己的训练模型，还是使用其他人的模型，我们都需要有关如何预处理训练数据的文档。作为作者，
# 其工作是编写并以某种合理的格式提供该文档。作为采用者，其工作是在准备数据时找到这些信息并遵循它。
#-----------------------------------------------------------第二部分--------------------------------------------
 def  make_model(number_of_layers=2, neurons_per_layer=32,dropout_ratio=0.2, optimizer='adam'):
   model = tf.keras.models.Sequential()
   model.add(tf.keras.models.Dense(neurons_per_layer,
   input_shape=[number_of_pixels],
   activation='relu', kernel_constraint=tf.keras.models.maxnorm(3)))
   model.add(tf.keras.models.Dropout(dropout_ratio))
   for i in range(number_of_layers-1):
     model.add(tf.keras.models.Dense(neurons_per_layer,
     activation='relu',
     kernel_constraint=tf.keras.models.maxnorm(3)))
     model.add(tf.keras.models.Dropout(dropout_ratio))
     model.add(tf.keras.models.Dense(number_of_classes, activation='softmax'))
     model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
   return model


from keras.wrappers.scikit_learn import KerasClassifier
#当scikit-learn调用make_model()时，它将为函数的参数赋予我们在创建KerasClassifier时提供的值。
#KerasClassifier:包装器
#  实际上，包装器只接收我们提供给它的值，并将它们传递给同名的模型制作函数参数。，如果没用分配值，
#将使用自身的默认参数
#KerasClassifier()的最后3个参数（epochs、batch_size和verbose）不是给模型的，而是给scikit-learn的。
# 它们被传递给交叉验证器的fit()例程以控制训练过程。
kc_model = KerasClassifier(build_fn=make_model,
 number_of_layers=2, neurons_per_layer=32,
 optimizer = 'adam',
 epochs=100, batch_size=256, verbose=0)

from sklearn.model_selection import StratifiedKFold
#交叉验证器
kfold = StratifiedKFold(n_splits=10, shuffle=True,#
random_state=random_seed)
#记录分数
from sklearn.model_selection import cross_val_score
results = cross_val_score(kc_model, X_train, original_y_train,
 cv=kfold, verbose=0)
print('results = {}\nresults.mean = {}'.format(
 results, results.mean()))
#创建pipeline
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler
estimators = []
estimators.append(('normalize_step', MinMaxScaler()))
estimators.append(('model_step', kc_model))
pipeline = Pipeline(estimators)
#使用简写符号创建pipeline，每个步骤都没有名称。这两个pipeline对象是相同的。唯一的区别是，在第一个版本
#中，我们给每个步骤取了自己的名字。
pipeline = make_pipeline(MinMaxScaler(), kc_model)
#创建一个字典来搜索模型的3个参数：全连接层的数量（每个都有dropout），
# 全连接层的神经元的数量，以及两个不同的优化器。
param_grid = dict(model__number_of_layers=[ 2, 3, 4 ],
 model__neurons_per_layer=[ 20, 30, 40 ],
 model__optimizer=['adam', 'adadelta'])
#创建GridSearchCV对象，该对象将遍历参数网格，为每个选项组合组装一个模
#型，并交叉验证该模型
from sklearn.model_selection import GridSearchCV
grid_searcher = GridSearchCV(estimator=pipeline,
 param_grid=param_grid, verbose=2)
#search_results1中的一个对象是一个名为cv_results_的字典,cv_results_字典包含关于交叉验证结果的详细信息。
#“params”项告诉我们这组参数对应的每个分数。“mean_test_score”项告诉我们每组参数的交叉验证平均值。
search_results1 = grid_searcher.fit(X_train, original_y_train)
#输入层（函数式API需要的，784个元素）
input_layer = Input(shape=[784])
#形状为28×28×1的输入张量
input_layer = Input(shape=[28,28,1])
input = Input(shape=(784,))
dense_1 = Dense(1000, activation='relu')
dense_2 = Dense(500, activation='relu')
output = Dense(1, activation='sigmoid')
#构建连接层
C1_input = input
C1_dense_1 = dense_1(C1_input)
C1_dense_2 = dense_2(C1_dense_1)
C1_output = output(C1_dense_2)
#基于输入和输出连接层构建模型。其他层被隐含在内
network_1 = Model(C1_input, C1_output)

#  用一些来自第一个模型的网络层构建新模型
convo_1 = Conv2D(32, (5,5))
flatten_1 = Flatten()
# Build the new connection layers
C2_input = input
C2_dense_1 = dense_1(C2_input)
C2_convo_1 = convo_1(C2_dense_1)
C2_flatten_1 = flatten_1(C2_convo_1)
C2_output = output(C2_flatten_1)
# build the model
model2 = Model(C2_input, C2_output)