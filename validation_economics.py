from keras.models import load_model
from sklearn.model_selection import train_test_split
from utils_economics import load_text_processor
from utils_economics import Seq2Seq_Inference
import pandas as pd

#read in data sample 2M rows (for speed of tutorial)
traindf, testdf = train_test_split(pd.read_csv('data/economics/Full-Economic-News-DFE-839861_unix2.csv'),
                                   test_size=.10)
body_text = testdf.text.tolist()
title_text = testdf.headline.tolist()
seq2seq_Model_glove = load_model('data/economics/seq2seq_model_tutorial_glove.hdf5')
seq2seq_Model_fasttext = load_model('data/economics/seq2seq_model_tutorial_fasttext.hdf5')
seq2seq_Model_word2vec = load_model('data/economics/seq2seq_model_tutorial_word2vec.hdf5')
#seq2seq_Model_custom = load_model('data/economics/seq2seq_model_tutorial_custom.hdf5')
num_encoder_tokens, body_pp = load_text_processor('data/economics/body_pp.dpkl')
num_decoder_tokens, title_pp = load_text_processor('data/economics/title_pp.dpkl')
seq2seq_inf_glove = Seq2Seq_Inference(encoder_preprocessor=body_pp,
                                 decoder_preprocessor=title_pp,
                                 seq2seq_model=seq2seq_Model_glove)
# this method displays the predictions on random rows of the holdout set
#seq2seq_inf_glove.demo_model_predictions(n=10, issue_df=testdf)


bleu = seq2seq_inf_glove.evaluate_model(body_text[:10000], title_text[:10000])
print(f"\n****** Glove BLEU scrore ******:\n {bleu}")

seq2seq_inf_fasttext = Seq2Seq_Inference(encoder_preprocessor=body_pp,
                                 decoder_preprocessor=title_pp,
                                 seq2seq_model=seq2seq_Model_fasttext)
# this method displays the predictions on random rows of the holdout set
#seq2seq_inf_fasttext.demo_model_predictions(n=10, issue_df=testdf)


bleu = seq2seq_inf_fasttext.evaluate_model(body_text[:10000], title_text[:10000])
print(f"\n****** Fasttext BLEU scrore ******:\n {bleu}")

seq2seq_inf_word2vec = Seq2Seq_Inference(encoder_preprocessor=body_pp,
                                 decoder_preprocessor=title_pp,
                                 seq2seq_model=seq2seq_Model_word2vec)
# this method displays the predictions on random rows of the holdout set
#seq2seq_inf_word2vec.demo_model_predictions(n=10, issue_df=testdf)


bleu = seq2seq_inf_word2vec.evaluate_model(body_text[:10000], title_text[:10000])
print(f"\n****** Word2vec BLEU scrore ******:\n {bleu}")

#seq2seq_inf_custom = Seq2Seq_Inference(encoder_preprocessor=body_pp,
#                                 decoder_preprocessor=title_pp,
#                                 seq2seq_model=seq2seq_Model_custom)
# this method displays the predictions on random rows of the holdout set
#seq2seq_inf_custom.demo_model_predictions(n=10, issue_df=testdf)


#bleu = seq2seq_inf_custom.evaluate_model(body_text[:10000], title_text[:10000])
#print(f"\n****** Csutom BLEU scrore ******:\n {bleu}")
