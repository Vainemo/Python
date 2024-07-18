from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import sys
window_step=5
#读取文本
def get_text(input_file):
 #打开对应文件
 file = open(input_file, 'r')
 text = file.read()
 file.close()
 #text = text.lower()
 # 去掉换行符
 text = text.replace('\n',' ')
 text = text.replace(' ', ' ')
 print('corpus length:', len(text))
 return text
# 通过切分输入文本为有重叠的片段来构建文本碎片，每个文本碎片有 window_ length个字符。
def build_fragments(text, window_length):
 # make overlapping fragments of window_length characters
 fragments = []
 targets = []
 for i in range(0, len(text)-window_length, window_step):
   fragments.append(text[i: i + window_length])
   targets.append(text[i + window_length])
 print('number of fragments of length window_length=',window_length,':', len(fragments))
 return (fragments, targets)
#构建一个字典将每个字符转换为唯一的数字，反之亦然。
def build_dictionaries(text):
 unique_chars = sorted(list(set(text)))
 print('total unique chars:', len(unique_chars))
 char_to_index =dict((ch, index) for index, ch in enumerate(unique_chars))
 index_to_char = dict((index, ch) for index, ch in enumerate(unique_chars))
 return (unique_chars, char_to_index, index_to_char)

#将片段和目标转化为名为X和y的独热版本。
def encode_training_data(fragments, window_length, targets,
 char_to_index, index_to_char):
 # Turn inputs and targets into one-hot versions
 X = np.zeros((len(fragments), window_length,len(char_to_index)), dtype=np.bool)
 y = np.zeros((len(fragments), len(char_to_index)),dtype=np.bool)
 for i, fragment in enumerate(fragments):
    for t, char in enumerate(fragment):
       X[i, t, char_to_index[char]] = 1
    y[i, char_to_index[targets[i]]] = 1
 return (X, y)


#两个LSTM层和唯一的全连接层
def build_model(window_length, num_unique_chars):
 model = Sequential()
 model.add(LSTM(128, return_sequences=True,
 input_shape=(window_length, num_unique_chars)))
 model.add(LSTM(128))
 model.add(Dense(num_unique_chars, activation='softmax'))
 optimizer = RMSprop(lr=0.01)
 model.compile(loss='categorical_crossentropy',
 optimizer=optimizer)
 return model

def generate_text(model, X, y, number_of_epochs, temperatures,index_to_char, char_to_index, file_writer):
 for iteration in range(number_of_epochs):
   print('--------------------------------------\n',file_writer)
   print('Iteration '+str(iteration)+'\n',file_writer)
   history = model.fit(X, y, batch_size=batch_size, epochs=1)
   start_index = random.randint(0, len(text)-window_length-1)
   for temperature in temperatures:
      #print('\n----- temperature: '+\str(temperature)+'\n',file_writer)
      seed = text[start_index: start_index+window_length]
      generated = seed
      print('----- Generating with seed:'\'<' +seed+'>\n',file_writer)
      for i in range(generated_text_length):
        x = np.zeros((1, window_length,len(index_to_char)))
        for t, char in enumerate(seed):
           x[0, t, char_to_index[char]] = 1.
        preds = model.predict(x, verbose=0)[0]
        next_index = choose_probability(preds,temperature)
        next_char = index_to_char[next_index]
        generated += next_char
        seed = seed[1:] + next_char
      print(generated+'\n\n', file_writer)
      file_writer.flush()