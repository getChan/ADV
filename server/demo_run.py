import tensorflow as tf
# import matplotlib.pyplot as plt
import sentencepiece as spm
from transformer import CustomSchedule, Transformer, create_masks
from time import time
print(tf.__version__)
print(tf.test.is_gpu_available())

sp = spm.SentencePieceProcessor()
sp.Load('hanmun_encode.model') 

OLD_VOCAB_SIZE = 26000
KOR_VOCAB_SIZE = 52000
# korean dictionary
with open('josun.vocab', encoding='utf-8') as f:
    vo = [doc.strip().split("\t") for doc in f]
    # w[0]: token name    
    # w[1]: token score
    dict_to_korean = {i:w[0] for i, w in enumerate(vo)}
    dict_to_korean[KOR_VOCAB_SIZE] = '<START>'
    dict_to_korean[KOR_VOCAB_SIZE+1] = '<END>'

with open('hanmun.vocab', encoding='utf-8') as f:
    vo = [doc.strip().split("\t") for doc in f]
    dict_to_code = {w[0]:i for i, w in enumerate(vo)}

def query_encode(query):
    # start = time()
    tmp =  [dict_to_code[ch] for ch in sp.EncodeAsPieces(query)]
    # print("subword tokenize time", time()-start)
    return tmp


def load_model(path='./model'):
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8

    added_input_size = OLD_VOCAB_SIZE + 2
    added_target_size = KOR_VOCAB_SIZE + 2
    dropout_rate = 0.1

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                          added_input_size, added_target_size, 
                          pe_input=added_input_size, 
                          pe_target=added_target_size,
                          rate=dropout_rate)

    checkpoint_path = path
    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print ('Model Load completed!!')
        return transformer
    else:
        print('Model Load Fail..')
        raise FileNotFoundError


model = load_model()

def evaluate(inp_sentence, transformer=model):

    # start = time()

    start_token = OLD_VOCAB_SIZE
    end_token = OLD_VOCAB_SIZE+1
    
    inp_sentence = [start_token] + inp_sentence + [end_token]
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [KOR_VOCAB_SIZE]
    output = tf.expand_dims(decoder_input, 0)
    
    for i in range(200):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input, 
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == KOR_VOCAB_SIZE+1:
            # print("model predict time :", time()-start)
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    # print("model predict time :", time()-start)
    return tf.squeeze(output, axis=0), attention_weights


def translate(sentence, plot=''):
    result, attention_weights = evaluate(query_encode(sentence), model)
    predicted_sentence = [dict_to_korean[x.numpy()] for x in result[1:]]

    # if plot:
    #     plot_attention_weights(attention_weights, sentence, result, plot)

    return ' '.join(predicted_sentence)

# def plot_attention_weights(attention, sentence, result, layer):
#     sentence = encode_original(sentence)
#     attention = tf.squeeze(attention[layer], axis=0)
#     for head in range(attention.shape[0]):
#         ax = fig.add_subplot(2, 4, head+1)

#         ax.matshow(attention[head][:-1, :])

#         ax.set_xticks(range(len(sentence)+2))
#         ax.set_yticks(range(len(result)))

#         ax.set_ylim(len(result)-1.5, -0.5)

#         ax.set_xticklabels(
#             ['<start>']+[reverse_input_id[i] for i in sentence]+['<end>'], rotation=90)

#         ax.set_yticklabels([reverse_input_id[i.numpy()] for i in result 
#                             if i < target_vocab_size])

#         ax.set_xlabel('Head {}'.format(head+1))

#     plt.tight_layout()
#     plt.show()