from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as keras_backend
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import numpy as np
random_seed = 42
np.random.seed(random_seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
image_height = X_train.shape[1]
image_width = X_train.shape[2]
#interp()函数将所有输入值从[0,255]转换到[-1,1]
X_train = np.interp(X_train, [0, 255], [-1,1])
X_test = np.interp(X_test, [0, 255], [-1,1])
#保存未被重塑之前的数据
original_y_train = y_train
original_y_test = y_test
#将数据转化为四维张量，即60000×1×28×28（重塑）
X_train = X_train.reshape(X_train.shape[0],image_height, image_width, 1)
X_test = X_test.reshape(X_test.shape[0],image_height, image_width, 1)
#-------构建卷积层-----------------
#构建一个包含15个3x3的过滤器的卷积层
#Conv2D第一个强制参数是一个整数，指定层应该管理的过滤器的数量。
#第二个强制参数是一个列表，用于提供该层上的过滤器的维数。
convolution_layer = Conv2D(15, (3, 3))
#activation：激活函数, strides：步幅,padding：零填充,默认字符是valid，意思是没有充填，sanme意思是使输出和输入的大小相同
#input_shape:当且仅当该层是网络中的第一层时需要这个参数，描述整个数据集的形状（28x28x1)
#model.add(convolution_layer, activation='relu',strides=(2, 2), padding='same',input_shape=(image_height, image_width, 1))
def make_simple_cnn_model():
 model = Sequential()
 model.add(Conv2D(32, (5, 5),
 activation='relu', padding='same',
 input_shape=(image_height, image_width, 1)))
 #平整层
 model.add(Flatten())
 #全连接层
 model.add(Dense(number_of_classes, activation='softmax'))
 #损失函数
 model.compile(loss='categorical_crossentropy',
 #优化器
 optimizer='adam',
 #返回准确率
 metrics=['accuracy'])
 return model

simple_cnn_model = make_simple_cnn_model()
simple_cnn_history = simple_cnn_model.fit(X_train, y_train,validation_data=(X_test, y_test),epochs=100, batch_size=256)
#使用了三个卷积层的模型，添加了dropout层
def make_bigger_cnn_model():
 model = Sequential()
 model.add(Conv2D(16, (5, 5), activation='relu',
 padding='same',
 #kernel_constraint:防止过滤器中的值变大(最大为3)
 kernel_constraint=maxnorm(3),
 input_shape=(image_height, image_width, 1)))
 model.add(Dropout(0.2))
 model.add(Conv2D(8, (3, 3), activation='relu', padding='same',
 kernel_constraint=maxnorm(3)))
 model.add(Dropout(0.2))
 model.add(Conv2D(8, (3, 3), activation='relu', padding='same',
 kernel_constraint=maxnorm(3)))
 model.add(Dropout(0.2))
 model.add(Flatten())
 model.add(Dense(number_of_classes, activation='softmax'))
 model.compile(loss='categorical_crossentropy',
 optimizer='adam',
 metrics=['accuracy'])
 return model
#创建模型
bigger_cnn_model = make_bigger_cnn_model()
bigger_cnn_history = bigger_cnn_model.fit(X_train, y_train,validation_data=(X_test, y_test),epochs=100, batch_size=256)
#添加了池化层的模型，降低过拟合
def make_pooling_cnn_model():
 model = Sequential()
 model.add(Conv2D(30, (5, 5), activation='relu',
 padding='same',
 kernel_constraint=maxnorm(3),
 input_shape=(image_height, image_width, 1)))
 model.add(Dropout(0.2))
 #最大池化层 步幅设置为2，2
 model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
 model.add(Conv2D(16, (3, 3), activation='relu',
 padding='same',
 kernel_constraint=maxnorm(3)))
 model.add(Dropout(0.2))
 model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
 model.add(Flatten())
 model.add(Dense(128, activation='relu'))
 model.add(Dense(64, activation='relu'))
 model.add(Dense(number_of_classes, activation='softmax'))
 model.compile(loss='categorical_crossentropy',
 optimizer='adam',
 metrics=['accuracy'])
 return model
#去掉最大池化层，而是将卷积层的步幅设置为（2，2），虽然稍微降低了准确率，但时间减少了75%
#（时间快的原因1.设置了过滤器步幅，使过滤器的频率更低。2.减少了池化层的操作）
def make_striding_cnn_model():
 model = Sequential()
 model.add(Conv2D(30, (5, 5), activation='relu',
 padding='same', strides=(2, 2),
 kernel_constraint=maxnorm(3),
 input_shape=(image_height, image_width, 1)))
 model.add(Dropout(0.2))
 model.add(Conv2D(16, (3, 3), activation='relu',
 padding='same', strides=(2, 2),
 kernel_constraint=maxnorm(3)))
 model.add(Dropout(0.2))
 model.add(Flatten())
 model.add(Dense(128, activation='relu'))
 model.add(Dense(64, activation='relu'))
 model.add(Dense(number_of_classes, activation='softmax'))
 model.compile(loss='categorical_crossentropy',
 optimizer='adam',
 metrics=['accuracy'])
 return model

#模型去掉了Dropout层，将批归一化层放在卷积层和激活函数之间
def make_striding_batchnorm_cnn_model():
 model = Sequential()
 model.add(Conv2D(30, (5, 5), activation=None,
 padding='same', strides=(2, 2),
 input_shape=(image_height, image_width, 1)))
 model.add(BatchNormalization())
 model.add(Activation('relu'))
 model.add(Conv2D(16, (3, 3), activation=None,
 padding='same', strides=(2, 2)))
 model.add(BatchNormalization())
 model.add(Activation('relu'))
 model.add(Flatten())
 model.add(Dense(128, activation='relu'))
 model.add(Dense(64, activation='relu'))
 model.add(Dense(number_of_classes, activation='softmax'))
 model.compile(loss='categorical_crossentropy',
 optimizer='adam',
 metrics=['accuracy'])
 return model

#图像生成器，使用一个ImageDataGenerator来产生转换过的图像。每幅图都可能被随机水平
#翻转过，或者向某个方向被旋转了至多100°。
image_generator = ImageDataGenerator(
 rotation_range=100,
 horizontal_flip=True)
#使用fit_generator()而不是fit()来开始训练，fit_generator()第一个参数是一个返回批量样本的函数
#send：此代码的每次运行都会产生不同的结果。对于测试和调试，每次都返回相同的序列通常很有用。因此我们可以通过在调用
#flow()时设置seed参数来强制执行此操作
#不同点：这里需要指定每个epochs的大小
#原因：只要我们继续调用生成器，它就会不断地产生变体（数据越来越多），因此遍历通常称为一个epoch的“所有数据”
#并不会有真正意义。例如，统计信息是在一个epoch结束时被采集的，我们的回调函数也是此时被调用的。因此
#我们告诉flow()在它简单地声明一个epoch结束之前要生成多少张图像
model.fit_generator(image_generator.flow(
 X_train, y_train, batch_size=256),
 seed=42, epochs=100,
 samples_per_epoch=len(X_train))

