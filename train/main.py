import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
import time
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from functions import *
warnings.filterwarnings(action='ignore')
df = pd.read_hdf('../data/tokenized_10thousand.hdf')

# json으로 저장한 단어 사전 불러오기
with open('../notebook/id_dict/input_id.json', 'r') as fp:
    input_id = json.load(fp)

with open('../notebook/id_dict/target_id.json', 'r') as fp:
    target_id = json.load(fp)

#  id->원래 단어 역변환
reverse_input_id = {i:char for char, i in input_id.items()}
reverse_input_id[1] = '_'
reverse_target_id = {i:char for char, i in target_id.items()}
reverse_target_id[1] = '_'

origin, trans = df.original.apply(encode_original), df.translation.apply(encode_translation)

encoder_input_data = tf.convert_to_tensor(pad_sequences(origin, maxlen=200, padding='post', truncating='post'),dtype=tf.int64)
decoder_target_data = tf.convert_to_tensor(pad_sequences(trans, maxlen=200, padding='post', truncating='post'),dtype=tf.int64)

train_dataset = tf.data.Dataset.from_tensor_slices((encoder_input_data, decoder_target_data))

BUFFER_SIZE = 20000
BATCH_SIZE = 64

train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = len(input_id) + 2
target_vocab_size = len(target_id) + 2
dropout_rate = 0.1

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, 
                                     True, 
                                     enc_padding_mask, 
                                     combined_mask, 
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

EPOCHS = 20

for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> original(chinese), tar -> target(korean)
    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

        if batch % 50 == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
              epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                             ckpt_save_path))

    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))

    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))