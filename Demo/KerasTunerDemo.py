#----------------使用KerasTuner调整超参数-------------------------------
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

#加载数据
(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()
#归一化数据
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0
#定义模型（构建需要调参的模型时，需要这一步）
def Model_builder(hp):
    model=keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    #hp.Int 是 Keras Tuner 库中用于定义整数型超参数的函数
    #1.name：超参数的名称，这里是 'units'
    #2.min_value：超参数的最小值，这里是 32。
    #3.max_value：超参数的最大值，这里是 512。
    #4.step：超参数的步长，每次增加的值，这里是 32。
    hp_units=hp.Int('units',min_value=32,max_value=512,step=32)
    model.add(keras.layers.Dense(units=hp_units,activatiob='relu'))
    model.add(keras.layers.Dense(10))
    #choice函数 （hp.Choice 可以定义一个超参数，并指定它可以取的一组离散值，Keras Tuner 会在这些值中选择最优值，以优化模型的性能）。
    # name：超参数的名称（例如 'learning_rate'）。
    # values：该超参数的可选值列表（例如 [32, 64, 128]）。
    hp_learning_rate=hp.choice('learning_rate',values=[1e-2,1e-3,1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
#调节器：Keras Tuner 提供了四种调节器
  #1.RandomSearch 2.Hyperband 3.BayesianOptimization 4.Sklearn

#实例化Hyperband调节器：Hyperband 算法通过分阶段调整训练预算，能够更快地找到最优超参数组合
#1.第一个参数：定义了要调优的模型
#2.第二个参数(objective)：定义了要优化的目标。可以是模型的任意度量指标
#3.第三个参数(max_epochs)：每个模型训练的最大轮数
#4.第四个参数(factor):这是 Hyperband 的一个控制参数，决定了资源分配的减少率。每一轮中，Hyperband 会将资源减少到原来的 1/factor。
# 在这个例子中，factor 是 3，这意味着每次减少到原来的三分之一。
#5.第五个参数(directory)：用于保存搜索结果的目录。在这个例子中，结果将保存到 'my_dir' 目录。
#6.第六个参数(project_name)：项目名称，用于区分不同的调优任务。在这个例子中，项目名称是 'intro_to_kt'。
# Keras Tuner 会在 directory 中创建一个以 project_name 命名的子目录来保存特定调优任务的结果。
tuner=kt.Hyperband(Model_builder,objective='val_accuracy',max_epochs=10,factor=3,directory='my_dir',project_name='intro_to_kt')
#Hyperband算法原理：
#  #Hyperband 算法通过将训练资源（如训练轮数）分配给多个模型，并在多个阶段中进行筛选，逐步淘汰性能较差的模型，
# 从而高效地找到最优超参数组合。具体来说：
#1.初始阶段：Hyperband 会为大量模型分配较少的训练资源（如训练轮数）。
#2.筛选阶段：根据目标指标（如验证准确率），保留表现较好的模型，并增加它们的训练资源。不好的模型减少训练资源（factor参数管控）
#3.多轮筛选：重复上述过程，逐步减少模型数量，增加训练资源，直到达到最大训练轮数或预设的资源预算。

#创建回调以在验证损失达到特定值后提前停止训练。
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

#运行超参数搜索
tuner.search(img_train,label_train,epochs=50,validation_split=0.2,callbacks=[stop_early])
#h获取最好的模型
#num_trials：这个参数指定返回多少个最优的超参数配置。在这个例子中，num_trials=1 表示只返回一个最优配置。
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

#训练模型
model=tuner.hypermodel.build(best_hps)
history=model.fit(img_train,label_train, epochs=50, validation_split=0.2)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
#重新实例化超模型并使用上面的最佳周期
hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)
#评估模型
eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)

 