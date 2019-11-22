import json
def chinese_to_id(chinese):
    with open('../../prototype/id_dict/input_id.json', 'r') as fp:
        input_id = json.load(fp)
    return input_id[chinese]

def korean_to_id(korean):
    with open('../../prototype/id_dict/target_id.json', 'r') as fp:
        target_id = json.load(fp)
    return target_id[korean]
    
def id_to_korean(id):
    with open('../../prototype/id_dict/reversed_target_id.json', 'r') as fp:
        reversed_target_id = json.load(fp)
    return reversed_target_id[id]
    
def id_to_chinese(id):
    with open('../../prototype/id_dict/reversed_input_id.json', 'r') as fp:
        reversed_input_id = json.load(fp)
    return reversed_input_id[id]
    
def padding_input(sequence):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(sequence, maxlen=200, padding='post', truncating='post')