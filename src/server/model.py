from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
class MTModel(object):
    '''
    seq2seq 모델 클래스
    '''
    def __init__(self):
        '''
        set hyperparameter
        '''
        self.latent_dim = 256  # LSTM cell dimension
    
    def load_saved_model(self, modelpath):
        '''
        저장된 모델을 불러와 파라미터를 초기화합니다.
        modelpath : 모델 파일의 경로
        '''
        model = load_model(modelpath)

        encoder_inputs = model.input[0]   # input_1
        encoder_embedding = model.layers[2](encoder_inputs)
        encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output   # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = model.input[1]   # input_2
        decoder_embedding = model.layers[3](decoder_inputs)
        decoder_state_input_h = Input(shape=(self.latent_dim,), name='input_3')
        decoder_state_input_c = Input(shape=(self.latent_dim,), name='input_4')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[5]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_embedding, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.layers[6]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)