#-----对服装图像进行分类--------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#加载fashion_mnist数据集每张照片是28*28像素大小，深度在0-255之间的服饰图片，共60000张
fashion_mnist=tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_lavles)=fashion_mnist.load_data()
#数据集中共有10个标签，先将标签与名称对应
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#进行归一化处理
train_images = train_images / 255.0
test_images = test_images / 255.0
#matplotlib.pyplot:用于绘制各种类型的图表
#figure:创建一个新图形，大小为10*10
plt.figure(figsize=(10,10))

for i in range(25):
    #subplot在一个图形中创建多个子图。第五行第五列的第i+个子图
    plt.subplot(5,5,i+1)
    #设置 x 轴和 y 轴的刻度
    plt.xticks([])
    plt.yticks([])
    #不显示网格
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
# plt.show()

model=tf.keras.Sequential([
    #平整层，将数据从二维数据转换为一维数据
    tf.keras.layers.Flatten(input_shape=(28,28)),
    #两个全连接层，第一个有128个神经元，第二个有10个，给出10个输出
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10)
])
#设定优化器，损失函数和指标
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

#开始训练
model.fit(train_images,train_labels,epochs=10)

#评估准确率，比较模型在测试数据集上的表现
test_loss,test_acc=model.evaluate(test_images,test_lavles,verbose=2)
print('测试集准确率：',test_acc)

#进行预测
#Softmax:归一化指数函数可以把输出项的得分变成概率
probability_model=tf.keras.Sequential([model,tf.keras.layers.Softmax()])
#predictions为所有测试集的预测结果
predictions=probability_model.predict(test_images)


#使用模型
img = test_images[1]
#tf.keras 模型经过了优化，可同时对一个批或一组样本进行预测。因此，即便您只使用一个图像，您也需要将其添加到列表中
img = (np.expand_dims(img,0))
#keras.Model.predict 会返回一组列表，每个列表对应一批数据中的每个图像。在批次中获取对我们（唯一）图像的预测：
predictions_single = probability_model.predict(img)
print(predictions_single)
np.argmax(predictions_single[0])