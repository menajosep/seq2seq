from utils import load_decoder_inputs, load_encoder_inputs, load_text_processor
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, Bidirectional, BatchNormalization
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import numpy as np
from gensim.models import KeyedVectors

encoder_input_data, doc_length = load_encoder_inputs('data/recipes/train_body_vecs.npy')
decoder_input_data, decoder_target_data = load_decoder_inputs('data/recipes/train_title_vecs.npy')

num_encoder_tokens, body_pp = load_text_processor('data/recipes/body_pp.dpkl')
num_decoder_tokens, title_pp = load_text_processor('data/recipes/title_pp.dpkl')

#arbitrarly set latent dimension for embedding and hidden units
latent_dim = 300

# load glove embeddings
emb_file = '/home/jmena/dev/data/word2vec/GoogleNews-vectors-negative300.bin'
embeddings = KeyedVectors.load_word2vec_format(emb_file, binary=True)

# build encoder embedding matrix
encoder_embedding_matrix = np.zeros((num_encoder_tokens, latent_dim))
not_found = 0
print('Found %s word vectors.' % len(len(embeddings.vectors)))
for i, word in body_pp.id2token.items():
    embedding_index = embeddings.vocab[word].index
    embedding_vector = embeddings.vectors[embedding_index]
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        encoder_embedding_matrix[i] = embedding_vector
    else:
        not_found += 1
        print('%s word out of the vocab.' % word)

print('Found %s word out of the vocab.' % str(not_found))
##### Define Model Architecture ######

########################
#### Encoder Model ####
encoder_inputs = Input(shape=(doc_length,), name='Encoder-Input')

# Word embeding for encoder (ex: Issue Body)
x = Embedding(num_encoder_tokens, latent_dim, name='Body-Word-Embedding',
              mask_zero=False, weights=[encoder_embedding_matrix],
              trainable=False)(encoder_inputs)
x = BatchNormalization(name='Encoder-Batchnorm-1')(x)

# Intermediate GRU layer (optional)
#x = GRU(latent_dim, name='Encoder-Intermediate-GRU', return_sequences=True)(x)
#x = BatchNormalization(name='Encoder-Batchnorm-2')(x)

# We do not need the `encoder_output` just the hidden state.
_, state_h = GRU(latent_dim, return_state=True, name='Encoder-Last-GRU')(x)

# Encapsulate the encoder as a separate entity so we can just
#  encode without decoding if we want to.
encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')

seq2seq_encoder_out = encoder_model(encoder_inputs)

# build encoder embedding matrix
decoder_embedding_matrix = np.zeros((num_decoder_tokens, latent_dim))
print('Found %s word vectors.' % len(embeddings_index))
for i, word in title_pp.id2token.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        decoder_embedding_matrix[i] = embedding_vector

########################
#### Decoder Model ####
decoder_inputs = Input(shape=(None,), name='Decoder-Input')  # for teacher forcing

# Word Embedding For Decoder (ex: Issue Titles)
dec_emb = Embedding(num_decoder_tokens, latent_dim, name='Decoder-Word-Embedding',
                    mask_zero=False, weights=[decoder_embedding_matrix],
                    trainable=False)(decoder_inputs)
dec_bn = BatchNormalization(name='Decoder-Batchnorm-1')(dec_emb)

# Set up the decoder, using `decoder_state_input` as initial state.
decoder_gru = GRU(latent_dim, return_state=True, return_sequences=True, name='Decoder-GRU')
decoder_gru_output, _ = decoder_gru(dec_bn, initial_state=seq2seq_encoder_out)
x = BatchNormalization(name='Decoder-Batchnorm-2')(decoder_gru_output)

# Dense layer for prediction
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='Final-Output-Dense')
decoder_outputs = decoder_dense(x)

########################
#### Seq2Seq Model ####

#seq2seq_decoder_out = decoder_model([decoder_inputs, seq2seq_encoder_out])
seq2seq_Model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')

script_name_base = 'tutorial_seq2seq'

model_checkpoint = ModelCheckpoint('data/recipes/{:}.epoch{{epoch:02d}}-val{{val_loss:.5f}}_word2vec.hdf5'.format(script_name_base),
                                   save_best_only=True)

batch_size = 1200
epochs = 100
history = seq2seq_Model.fit([encoder_input_data, decoder_input_data], np.expand_dims(decoder_target_data, -1),
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.12, callbacks=[model_checkpoint])

#save model
seq2seq_Model.save('data/recipes/seq2seq_model_tutorial_word2vec.hdf5')
