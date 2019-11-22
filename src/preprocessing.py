import json
import numpy as np
def chinese_to_id(chinese):
    seq = np.array([ch for ch in chinese if ch !=' '])
    with open('../../prototype/id_dict/input_id.json', 'r') as fp:
        input_id = json.load(fp)
    seq = list(map(lambda x:input_id.setdefault(x,1), seq))
    return seq

def korean_to_id(korean):
    with open('../../prototype/id_dict/target_id.json', 'r') as fp:
        target_id = json.load(fp)
    return target_id[korean]
    
def id_to_korean(id):
    with open('../../prototype/id_dict/reversed_target_id.json', 'r') as fp:
        reversed_target_id = json.load(fp)
    return reversed_target_id[str(id)]
    
def id_to_chinese(id):
    with open('../../prototype/id_dict/reversed_input_id.json', 'r') as fp:
        reversed_input_id = json.load(fp)
    return reversed_input_id[str(id)]
    
def padding_input(sequence):
    # print('-------------------')
    # print(sequence)
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(sequence, maxlen=200, padding='post', truncating='post')