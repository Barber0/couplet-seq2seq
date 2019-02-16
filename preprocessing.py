import numpy as np
import gc
from gensim.corpora import Dictionary
gc.collect()

LINE = 100000
START = '<s>'
STOP = '</s>'
IN_NAME = 'train/in.txt'
OUT_NAME = 'train/out.txt'
DIC_NAME = 'result/dictionary.dict'
EXTRA = '*'


def get_in(add_symbol=False):
    txt_in = []
    with open(IN_NAME,'r') as f:
        for i in range(LINE):
            ln = f.readline()[:-1]
            if add_symbol:
                ln += ' '+STOP
            txt_in.append(ln.split())
    return txt_in

def get_out(add_symbol=False):
    txt_out = []
    with open(OUT_NAME,'r') as f:
        for i in range(LINE):
            ln = f.readline()[:-1]
            if add_symbol:
                ln = START+' '+ln+' '+STOP
            txt_out.append(ln.split())
            # txt_out.append(ln.split())
    return txt_out

def create_dic():
    txt_in = get_in()
    txt_out = get_out()
    dic = Dictionary(txt_in+txt_out+[[EXTRA]])
    dic.add_documents([[START,STOP,EXTRA]])
    dic.save(DIC_NAME)
    return dic
