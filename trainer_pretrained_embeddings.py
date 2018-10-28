from utils import load_decoder_inputs, load_encoder_inputs, load_text_processor
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, Bidirectional, BatchNormalization
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import numpy as np
import argparse


def main(emb_file, datasource, n_epochs, emb_type, learning_rate):

    encoder_input_data, doc_length = load_encoder_inputs('data/{}/train_body_vecs.npy'.format(datasource))
    decoder_input_data, decoder_target_data = load_decoder_inputs('data/{}/train_title_vecs.npy'.format(datasource))

    num_encoder_tokens, body_pp = load_text_processor('data/{}/body_pp.dpkl'.format(datasource))
    num_decoder_tokens, title_pp = load_text_processor('data/{}/title_pp.dpkl'.format(datasource))

    vocabulary = np.load('data/{}/words.dat'.format(datasource))

    #arbitrarly set latent dimension for embedding and hidden units
    latent_dim = 300

    # load embeddings
    embeddings_index = {}
    f = open(emb_file)
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except ValueError as ve:
            print(values)
    f.close()

    # build encoder embedding matrix
    encoder_embedding_matrix = np.zeros((num_encoder_tokens, latent_dim))
    not_found = 0
    print('Found %s word vectors.' % len(embeddings_index))
    for i, word in body_pp.id2token.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and word in vocabulary:
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
        if embedding_vector is not None and word in vocabulary:
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


    seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=learning_rate), loss='sparse_categorical_crossentropy')

    script_name_base = 'tutorial_seq2seq'

    model_checkpoint = ModelCheckpoint('data/{}/{:}.epoch{{epoch:02d}}-val{{val_loss:.5f}}_{}.hdf5'.format(datasource,
                                                                                                           script_name_base,
                                                                                                           emb_type),
                                       save_best_only=True)

    batch_size = 1024
    epochs = n_epochs
    history = seq2seq_Model.fit([encoder_input_data, decoder_input_data], np.expand_dims(decoder_target_data, -1),
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.12, callbacks=[model_checkpoint])

    #save model
    seq2seq_Model.save('data/{}/seq2seq_model_tutorial_{}.hdf5'.format(datasource,emb_type))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_file', type=str, default=None, help='file with embeddings')
    parser.add_argument('--emb_type', type=str, default=None, help='type of embeddings')
    parser.add_argument('--datasource', type=str, default=None, help='type of datasource recipes|kp20k|economics')
    parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learnng rate')
    args = parser.parse_args()
    main(emb_file=args.emb_file, datasource=args.datasource,
         n_epochs=args.n_epochs, emb_type=args.emb_type,
         learning_rate=args.learning_rate)
