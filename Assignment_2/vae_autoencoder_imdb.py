from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from model import VAE
import numpy as np
import os

class VAE(object):
    def create(self, vocab_size=500, max_length=300, latent_rep_size=200):
        self.encoder = None
        self.decoder = None
        self.sentiment_predictor = None
        self.autoencoder = None

        x = Input(shape=(max_length,))
        x_embed = Embedding(vocab_size, 64, input_length=max_length)(x)

        vae_loss, encoded = self._build_encoder(x_embed, latent_rep_size=latent_rep_size, max_length=max_length)
        self.encoder = Model(inputs=x, outputs=encoded)

        encoded_input = Input(shape=(latent_rep_size,))
        predicted_sentiment = self._build_sentiment_predictor(encoded_input)
        self.sentiment_predictor = Model(encoded_input, predicted_sentiment)

        decoded = self._build_decoder(encoded_input, vocab_size, max_length)
        self.decoder = Model(encoded_input, decoded)

        self.autoencoder = Model(inputs=x, outputs=[self._build_decoder(encoded, vocab_size, max_length), self._build_sentiment_predictor(encoded)])
        self.autoencoder.compile(optimizer='Adam',
                                 loss=[vae_loss, 'binary_crossentropy'],
                                 metrics=['accuracy'])

	def _build_encoder(self, x, latent_rep_size=200, max_length=300, epsilon_std=0.01):
		    h = Bidirectional(LSTM(500, return_sequences=True, name='lstm_1'), merge_mode='concat')(x)
		    h = Bidirectional(LSTM(500, return_sequences=False, name='lstm_2'), merge_mode='concat')(h)
		    h = Dense(435, activation='relu', name='dense_1')(h)

	    def sampling(args):
	        z_mean_, z_log_var_ = args
	        batch_size = K.shape(z_mean_)[0]
	        epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev=epsilon_std)
	        return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

	    z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
	    z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

	    def vae_loss(x, x_decoded_mean):
	        x = K.flatten(x)
	        x_decoded_mean = K.flatten(x_decoded_mean)
	        xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
	        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	        return xent_loss + kl_loss

	    return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def _build_decoder(self, encoded, vocab_size, max_length):
	    repeated_context = RepeatVector(max_length)(encoded)

	    h = LSTM(500, return_sequences=True, name='dec_lstm_1')(repeated_context)
	    h = LSTM(500, return_sequences=True, name='dec_lstm_2')(h)

	    decoded = TimeDistributed(Dense(vocab_size, activation='softmax'), name='decoded_mean')(h)

	    return decoded

	def _build_sentiment_predictor(self, encoded):
	    h = Dense(100, activation='linear')(encoded)

	    return Dense(1, activation='sigmoid', name='pred')(h)

MAX_LENGTH = 300
NUM_WORDS = 1000

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
