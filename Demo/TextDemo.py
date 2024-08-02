#-----------------------------------电影评论文本分类----------------------------------------
#将评论分为负面评论和正面评论
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
#下载评论文本文件
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
#1.aclImdb_v1：文件名 
#2.untar：是否需要解压
#3.cache_dir：存储缓存文件的目录。如果为 None，则使用默认的缓存目录。
#4.cache_subdir：存储缓存文件的子目录。如果为 ''，则文件会直接存储在 cache_dir 中。
dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
#链接路径
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
train_dir=os.path.join(dataset_dir,"tranin");
#返回一个包含目录内容的列表
os.listdir(train_dir)
#移除unsup目录
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
batch_size = 32
seed = 42
#拆分训练集，一部分作为验证集
#subset:表示当前数据是训练集还是验证集。
#validation_split: 用于验证的比例。如果为 None，则不进行验证分割。validation_split=0.2, 表示将数据集中的 20% 用作验证集
#shuffle=False 意味着在加载数据时不对数据进行随机打乱。这样可以保持数据的顺序性，
raw_train_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='training', 
    seed=seed)
raw_val_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)
raw_test_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/test', batch_size=batch_size)
def custom_standardization(input_data):
    #全部转化为小写
    lowercase=tf.strings.lower(input_data)
    strpped_html=tf.strings.regex_replace(lowercase,'<br />')
    #替换掉html的标记<br />，对数据进行预处理
    return tf.strings.regex_replace(strpped_html,'[%s]'%re.escape(string.punctuation,''))

max_features = 10000
sequence_length = 250
#创建一个 TextVectorization 层（拆分层）。它将文本转换为数字形式，
# 以便可以输入到模型中进行训练。它支持多种功能，包括标准化、分词、向量化等。
#max_tokens：词汇表中的最大令牌数。max_tokens=10000 表示限制词汇表大小为 10,000。词汇表大小是一个整数值，表示你希望处理的不同词汇的总数
#output_mode：向量化的模式，
   #1.'int': 返回整数索引（词汇表的 ID）
   #2.'binary': 返回二进制矩阵（每个令牌是否存在的布尔值）。
   #3.'count': 返回词频矩阵（每个令牌的出现次数）。
   #4.'tf-idf': 返回 TF-IDF 矩阵（词频-逆文档频率）
#output_sequence_length:生成的序列的固定长度。文本将被填充或截断到这个长度。如果为 None，
#则序列的长度会根据输入的实际长度变化。output_sequence_length=100 表示将所有文本序列标准化为长度 100。
vectorize_layer =layers.TextVectorization(standardize=custom_standardization,max_tokens=max_features,output_mode='int',
                                          output_sequence_length=sequence_length)
train_text = raw_train_ds.map(lambda x, y: x)
#adapt(train_text)将文本数据转换为数字索引，输入到模型中进行训练。
vectorize_layer.adapt(train_text)
#查看处理结果
def vectorize_text(text,lable):
    text=tf.expand_dims(text,-1)
    return vectorize_layer(text),lable
#通过迭代器获取raw_train_ds中的第一个数据
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))
#预处理训练集，验证集，测试集
train_ds=raw_train_ds.map(vectorize_text)
val_ds=raw_val_ds.map(vectorize_text)
test_ds=raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE
#从磁盘加载后，.cache() 会将数据保存在内存中。这将确保数据集在训练模型时不会成为瓶颈。
# 如果您的数据集太大而无法放入内存，也可以使用此方法创建高性能的磁盘缓存，这比许多小文件
# 的读取效率更高。
#prefetch() 会在训练时将数据预处理和模型执行重叠。
train_ds=train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds=val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds=test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#创建模型
embedding_dim = 16
model=tf.keras.Sequential([
    layers.Embedding(max_features+1,embedding_dim),
    layers.dropout(0,2)
    #平均池化层：通过对序列维度求平均值来为每个样本返回一个定长输出向量。这允许模型以尽可能最简单的方式处理变长输入。
    layers.GlobalAveragePoolingID(),
    layers.Dropout(0,2),
    layers.Dense(1)
    ])
#输出层的结构
model.summary()
#binary_crossentropy：它能够度量概率分布之间的“距离”，或者在我们的示例中，指的是度量 ground-truth 分布与预测值之间的“距离”
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

#训练模型：
epochs = 10
#model.fit() 会返回包含一个字典的 History 对象。该字典包含训练过程中产生的所有信息：
history=model.fit(train_ds,validation_data=val_ds,epochs=epochs)
#评估模型
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

history_dict = history.history
history_dict.keys()
#四个条目分别代表训练集准确率，验证机准确率，训练集损失，验证集损失
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

#导出模型
#创建一个新模型
export_model = tf.keras.Sequential([
#在模型内包含TextVectorization层，这样新数据输入时就不需要对相关格式进行预处理
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])
export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

#使用新数据进行推断：调用predict()方法
examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]

export_model.predict(examples)