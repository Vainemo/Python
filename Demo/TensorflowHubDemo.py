#------使用 Tensorflow Hub 和 Keras 进行迁移学习--------
#将影评分为积极（positive）或消极（nagetive）两类。这是一个二元（binary）或者二分类问题，一种重要且应用广泛的机器学习问题
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("版本号: ", tf.__version__)
print("Eager 模式: ", tf.executing_eagerly())
print("Hub 版本: ", hub.__version__)
print("GPU是否可用：", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
train_data, validation_data, test_data = tfds.load(name="imdb_reviews", split=('train[:60%]', 'train[60%:]', 'test'),
 as_supervised=True)
#前10个样本
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
 #预训练文本嵌入向量模型
embedding="https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer=hub.KarasLayer(embedding,input_shape=[],dtype=tf.string,trainable=True)
hub_layer(train_examples_batch[:3])
model = tf.keras.Sequential()

#----------------------构建模型-----------------------
#TensorFlow Hub 层：该层使用预训练的 SavedModel 将句子映射到其嵌入向量。
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary
#------------------------------------------------------

#----------------------训练模型-------------------------
#总训练次数=epochs*总批次数
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data,
                    verbose=1)
#------------------------------------------------------

#----------------------评估模型-------------------------
results=model.evaluate(test_data.batch(512),verbose=2)
for name,value in zip(model.metrics_names,results):
    print("%s:%.3f"%(name,value))