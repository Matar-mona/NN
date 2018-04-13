from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import imdb

MAX_LENGTH = 300
NUM_WORDS = 1000
vocab_size = 500
latent_rep_size = 200

x = Input(shape=(MAX_LENGTH,))
x_embed = Embedding(vocab_size, 64, input_length=MAX_LENGTH)(x)
encoder_h = Bidirectional(LSTM(500, return_sequences=True, name='lstm_1'), merge_mode='concat')(x)
encoder_h = Bidirectional(LSTM(500, return_sequences=False, name='lstm_2'), merge_mode='concat')(encoder_h)
encoder_h = Dense(435, activation='relu', name='dense_1')(encoder_h)
z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(encoder_h)
z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(encoder_h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
return z_mean + K.exp(z_log_var / 2) * epsilon

def _build_sentiment_predictor(encoded):
    h = Dense(100, activation='linear')(encoded)

    return Dense(1, activation='sigmoid', name='pred')(h)

encoded_input = Input(shape=(latent_rep_size,))
predicted_sentiment = _build_sentiment_predictor(encoded_input)
sentiment_predictor = Model(encoded_input, predicted_sentiment)

repeated_context = RepeatVector(MAX_LENGTH)(encoded)
decoder_h = LSTM(500, return_sequences=True, name='dec_lstm_1')(repeated_context)
decoder_h = LSTM(500, return_sequences=True, name='dec_lstm_2')(decoder_h)
h_decoded = TimeDistributed(Dense(vocab_size, activation='softmax'), name='decoded_mean')(decoder_h)
x_decoded_mean = decoder_mean(h_decoded)
decoder = Model(encoded_input, h_decoded)

# Compute VAE loss
x = K.flatten(x)
x_decoded_mean = K.flatten(x_decoded_mean)
xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = xent_loss + kl_loss

autoencoder = Model(x, x_decoded_mean)
autoencoder.compile(optimizer='Adam',
                         loss=[vae_loss, 'binary_crossentropy'],
                         metrics=['accuracy'])

#IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

print("Training data")
print(X_train.shape)
print(y_train.shape)

print("Number of words:")
print(len(np.unique(np.hstack(X_train))))

X_train = pad_sequences(X_train, maxlen=MAX_LENGTH)
X_test = pad_sequences(X_test, maxlen=MAX_LENGTH)

train_indices = np.random.choice(np.arange(X_train.shape[0]), 2000, replace=False)
test_indices = np.random.choice(np.arange(X_test.shape[0]), 1000, replace=False)

X_train = X_train[train_indices]
y_train = y_train[train_indices]

X_test = X_test[test_indices]
y_test = y_test[test_indices]

temp = np.zeros((X_train.shape[0], MAX_LENGTH, NUM_WORDS))
temp[np.expand_dims(np.arange(X_train.shape[0]), axis=0).reshape(X_train.shape[0], 1), np.repeat(np.array([np.arange(MAX_LENGTH)]), X_train.shape[0], axis=0), X_train] = 1

X_train_one_hot = temp

temp = np.zeros((X_test.shape[0], MAX_LENGTH, NUM_WORDS))
temp[np.expand_dims(np.arange(X_test.shape[0]), axis=0).reshape(X_test.shape[0], 1), np.repeat(np.array([np.arange(MAX_LENGTH)]), X_test.shape[0], axis=0), X_test] = 1

x_test_one_hot = temp

model = VAE()
model.create(vocab_size=NUM_WORDS, max_length=MAX_LENGTH)

model.autoencoder.fit(x=X_train, y={'decoded_mean': X_train_one_hot, 'pred': y_train},
                      batch_size=10, epochs=50, callbacks=[checkpointer],
                      validation_data=(X_test, {'decoded_mean': x_test_one_hot, 'pred':  y_test}))
