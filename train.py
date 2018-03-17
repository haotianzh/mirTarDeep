import numpy as np
import random as rd
import time
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Dot
from keras.layers import Merge
from keras.layers import Dropout
from keras.layers import GRU
from keras.layers import Concatenate
import re
import random
import matplotlib.pyplot as plt

### mfe
base_dir = {}
base_index = {'A':1,'G':2,'C':3,'T':4}
embedding_size = 10
max_sequence_length = 100
max_mirna_length = 25
targets = []
mirnas = []
pesudo_mirnas = []
pesudo_mfes = []
real_mfes = []
def generate_data(file):
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            target,pesudo_mirna,microrna,pesudo_mfe,pesudo_p,real_mfe,real_p = line.split('\t')
            if len(target) > 100:
                continue
            targets.append(target)
            pesudo_mirnas.append(pesudo_mirna)
            mirnas.append(microrna)
            pesudo_mfes.append(float(pesudo_mfe))
            real_mfes.append(float(real_mfe))
     
def turn2seq(data):
    dic = ['A','G','C','T']
    str = ''
    for pos in data:
        if pos == 0:
            continue
        str += dic[pos-1]
    return str

def turn2numeric(data,mode):
    res = []
    for sequence in data:
        seq = []
        for char in sequence:
            seq.append(base_index[char])
        res.append(seq)
    if mode == 'mirna':
        return pad_sequences(res,maxlen=max_mirna_length)
    elif mode == 'target':
        return pad_sequences(res,maxlen=max_sequence_length)

#### build process ####
generate_data('T:/microRNA_3/mfe_p_0315.txt')
targets_len = [len(x) for x in targets]
mirnas_len = [len(x) for x in mirnas]
mti_counts = len(targets) # 7646
target = turn2numeric(targets,'target')
pesudomirna = turn2numeric(pesudo_mirnas,'mirna')
realmirna = turn2numeric(mirnas,'mirna')
dev_target = np.concatenate((target[:500],target[:500]),axis=0)
dev_mirna = np.concatenate((realmirna[:500],pesudomirna[:500]),axis=0)
dev_mfe = np.concatenate((real_mfes[:500],pesudo_mfes[:500]),axis=0)
dev_label =  np.zeros([1000,2])
dev_label[:500,0] = 1
dev_label[500:,1] = 1
target = np.concatenate((target[500:],target[500:]),axis=0)
mirna = np.concatenate((realmirna[500:],pesudomirna[500:]),axis=0)
mfe = np.concatenate((real_mfes[500:],pesudo_mfes[500:]),axis=0)
label = np.zeros([mti_counts*2-1000,2])
label[:mti_counts-500,0] = 1
label[mti_counts-500:,1] = 1
label = label.astype(np.float32)

# train data and test data split, 5-fold cross validation 
# train_xxx contains five subset for cross validation.
indices = list(range(2*mti_counts-1000))
random.shuffle(indices)
split_ratios = np.array([0,0.2,0.4,0.6,0.8,1.0])
split_index = list(split_ratios * (2 * mti_counts - 1000))
split_index = [int(x) for x in split_index]
train_target = []
train_mfe = []
train_mirna = []
train_label = []
test_target = []
test_mfe = []
test_mirna = []
test_label = []

for i in range(1,6):
    test_target.append(target[indices[split_index[i-1]:split_index[i]],:])
    train_target.append(target[indices[0:split_index[i-1]] + indices[split_index[i]:],:])
    
    test_mirna.append(mirna[indices[split_index[i-1]:split_index[i]],:])
    train_mirna.append(mirna[indices[0:split_index[i-1]] + indices[split_index[i]:],:])
    
    test_mfe.append(mfe[indices[split_index[i-1]:split_index[i]]])
    train_mfe.append(mfe[indices[0:split_index[i-1]] + indices[split_index[i]:]])
    
    test_label.append(label[indices[split_index[i-1]:split_index[i]],:])
    train_label.append(label[indices[0:split_index[i-1]] + indices[split_index[i]:],:])

## expand mfe dimensions
for mmm in range(5):
    train_mfe[mmm] = train_mfe[mmm][:,np.newaxis]
    test_mfe[mmm] = test_mfe[mmm][:,np.newaxis]
dev_mfe = dev_mfe[:,np.newaxis]


## build and compile stack RNN model
#
#  target -> base embedding -> GRU -> dropout -> GRU \
#                                                      -> dot -> concatenate(mfe) -> softmax  
#  mirna -> base embedding -> GRU -> dropout -> GRU  /
##
lstm_dim = 64
input_target = Input(shape=(100,),dtype='int8',name='targets')
target_model = Embedding(input_length=max_sequence_length,input_dim=5,output_dim=128)(input_target)
target_model = GRU(lstm_dim,return_sequences=True)(target_model)
target_model = Dropout(rate=0.4)(target_model)
target_model = GRU(lstm_dim)(target_model)
input_mirna = Input(shape=(25,),dtype='int8',name='mirnas')
mirna_model = Embedding(input_length=max_mirna_length,input_dim=5,output_dim=128)(input_mirna)
mirna_model = GRU(lstm_dim,return_sequences=True)(mirna_model)
mirna_model = Dropout(rate=0.4)(mirna_model) 
mirna_model = GRU(lstm_dim)(mirna_model)
x = Dot([1,1])([target_model,mirna_model])
input_mfe = Input(shape=(1,),dtype='float32',name='mfes')
x = keras.layers.concatenate([x,input_mfe])
yhat = Dense(2, activation='softmax')(x)
model = Model(inputs=[input_target,input_mirna,input_mfe],outputs=yhat)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy','mse'])
model.summary()

## training cross-validation model
for i in range(5):
    time1 = time.time()
    print('begin training %d model...'%(i))
    model.fit({'targets':train_target[i],'mirnas':train_mirna[i],'mfes':train_mfe[i]},train_label[i],batch_size=32,epochs=20,verbose=True)
    time2 = time.time()
    score = model.evaluate({'targets':test_target[i],'mirnas':test_mirna[i],'mfes':test_mfe[i]}, test_label[i], batch_size=128)
    print ('finished,total time:' + str(time2 -time1))
    model.save('your path')





