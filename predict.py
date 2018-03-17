from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
filename = sys.argv[1]
max_sequence_length = 100
max_mirna_length = 25
model_dir = 'models/'
models = []
print('loading models...')
for file in os.listdir(model_dir):
    if  os.path.isfile(model_dir + file):
        if file.split('.')[1] == 'h5':
            model_path = model_dir + file
            model = load_model(model_path)
            models.append(model)
print('finished')
if not models:
    raise Exception('No model found.')

def turn2numeric(data,mode):
    base_index = {'A':1,'G':2,'C':3,'T':4}
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

def predict(mirna,target,mfe):
    mirna = mirna.upper().replace('U','T')
    target = target.upper().replace('U','T')
    mirna = turn2numeric([mirna],"mirna")
    target = turn2numeric([target],"target")
    scores = []
    for model in models:
        score = model.predict({"targets":target,"mirnas":mirna,'mfes':np.array([mfe])})
        scores.append(score)
    final = sum(scores) / len(scores)
    # print(final)
    return final[:,0]

def read_file(file):
    with open(file) as f:
        data = f.readlines()
    for line in data:
        mtiname,mirna,target,mfe = line.split('\t')
        score = predict(mirna,target,mfe)
        print('%s : %f'%(mtiname,score))
if __name__ == '__main__':
    read_file(filename)



