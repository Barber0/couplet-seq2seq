import numpy as np
import keras.backend as K
import gc
import sys
from keras.layers import Dense,LSTM,Embedding,Input,GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from preprocessing import START,STOP,LINE,DIC_NAME,EXTRA,get_in,get_out
from gensim.corpora import Dictionary
gc.collect()
K.clear_session()

BATCH = 50
EPOCHS = 100
LR = 1e-4
DECAY = 1e-6
MODEL_NAME = 'result/model-190215.h5'

BATCH_NUM = int(np.ceil(float(LINE)/BATCH))

dic = Dictionary.load(DIC_NAME)

txt_in = get_in(True)
txt_out = get_out(True)

def get_max_len(txt):
    return max(list(map(lambda ln: len(ln),txt)))

MAX_INPUT_LEN = get_max_len(txt_in)
MAX_TARGET_LEN = get_max_len(txt_out)
VOCAB_NUM = len(dic.token2id.values())

LSTM_CELLS = 256
EBD_CELLS = 200

def get_seq(dic,tin,max_input_len,start,stop,extra):
    txt_in = tin.strip()
    seq = np.zeros([1,max_input_len])
    words = dic.token2id.keys()
    txt_len = len(txt_in)
    for i in range(txt_len):
        if txt_in[i] in words:
            seq[0,i] = dic.token2id[txt_in[i]]
            # print(txt_in[i],'\t',dic.token2id[txt_in[i]])
        else:
            seq[0,i] = dic.token2id[extra]
    seq[0,txt_len] = dic.token2id[stop]
    return seq

def gen_seq(dic,tin,tout,batch,batch_num,max_input_len,max_target_len):
    token2id = dic.token2id
    vocab_num = len(token2id.values())
    while True:
        for i in range(batch_num):
            e_in = np.zeros([batch,max_input_len])
            d_in = np.zeros([batch,max_target_len])
            d_out = np.zeros([batch,max_target_len,vocab_num])
            tin_b = tin[i*batch:(i+1)*batch]
            tout_b = tout[i*batch:(i+1)*batch]
            for j in range(batch):
                for k,w in enumerate(tin_b[j]):
                    if w in token2id.keys():
                        e_in[j,k] = token2id[w]
                    else:
                        e_in[i,k] = token2id[EXTRA]
                for k,w in enumerate(tout_b[j]):
                    if w in token2id.keys():
                        d_in[j,k] = token2id[w]
                        if k > 0:
                            d_out[j,k-1,token2id[w]] = 1
                    else:
                        d_in[j,k] = token2id[EXTRA]
                        if k > 0:
                            d_out[j,k-1,token2id[EXTRA]] = 1
            yield [e_in,d_in],d_out

e_ipt = Input(shape=[MAX_INPUT_LEN])
e_ebd = Embedding(input_dim=VOCAB_NUM,output_dim=EBD_CELLS)
e_lstm = LSTM(LSTM_CELLS,return_state=True)

e_ebd_o = e_ebd(e_ipt)
e_o = e_lstm(e_ebd_o)

d_ipt = Input(shape=[MAX_TARGET_LEN])
d_ebd = Embedding(input_dim=VOCAB_NUM,output_dim=EBD_CELLS)
d_lstm = LSTM(LSTM_CELLS,return_sequences=True,return_state=True)
d_fc = Dense(VOCAB_NUM,activation='softmax')

d_ebd_o = d_ebd(d_ipt)
d_lstm_o = d_lstm(d_ebd_o,initial_state=e_o[1:])
d_fc_o = d_fc(d_lstm_o[0])

model = Model([e_ipt,d_ipt],d_fc_o)
chpt = ModelCheckpoint(MODEL_NAME,save_weights_only=True)
adam = Adam(lr=LR,decay=DECAY)
model.compile(optimizer=adam,loss='categorical_crossentropy')

model.load_weights(MODEL_NAME)

encoder = Model(e_ipt,e_o[1:])

d_e_h = Input(shape=[LSTM_CELLS])
d_e_s = Input(shape=[LSTM_CELLS])
d_lstm_o = d_lstm(d_ebd_o,initial_state=[d_e_h,d_e_s])
d_fc_o = d_fc(d_lstm_o[0])

decoder = Model([d_ipt,d_e_h,d_e_s],d_fc_o)

def predict(a_ipt):
    e_ipt = get_seq(dic,a_ipt,MAX_INPUT_LEN,START,STOP,EXTRA)
    e_state = encoder.predict(e_ipt)
    txt_out = ''
    d_ipt = np.zeros([1,MAX_TARGET_LEN])
    d_ipt[0,0] = dic.token2id[START]
    for i in range(MAX_TARGET_LEN):
        d_o = decoder.predict([d_ipt]+e_state)
        idx = np.argmax(d_o[0,i,:])
        word = dic[idx]
        if word == '</s>':
            break
        txt_out += word
        d_ipt[0,i+1] = idx
    return txt_out

if len(sys.argv) > 1:
    a_ipt = sys.argv[1]
    print(predict(a_ipt))
else:
    print('请输入上联')

# model.fit_generator(gen_seq(
#     dic,
#     txt_in,
#     txt_out,
#     BATCH,
#     BATCH_NUM,
#     MAX_INPUT_LEN,
#     MAX_TARGET_LEN
# ),steps_per_epoch=BATCH_NUM,epochs=EPOCHS,callbacks=[chpt])