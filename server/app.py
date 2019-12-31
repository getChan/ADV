from model import MTModel
from preprocessing import chinese_to_id, id_to_korean, padding_input
import numpy as np
from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/demo', methods=['POST'])
def demo():
    req = request.json
    demo_input = req['input']
    demo_input = chinese_to_id(demo_input)
    padded = padding_input([demo_input])

    return decode_sequence(padded)

def decode_sequence(input_seq):
    model = MTModel()
    model.load_saved_model('../../prototype/model/s2s_10.h5')
    states_value = model.encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = 2 # 2:<START>

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = model.decoder_model.predict(
            [target_seq] + states_value)
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = id_to_korean(sampled_token_index)
        decoded_sentence += sampled_char

        if (sampled_char == 'PAD' or len(decoded_sentence) > 200): # padded_length : 200
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence

app.run(host='0.0.0.0') 