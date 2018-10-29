from keras.models import load_model
from sklearn.model_selection import train_test_split
from utils import load_text_processor
from utils_recipes import Seq2Seq_Inference
import pandas as pd

#read in data sample 2M rows (for speed of tutorial)
testdf = pd.read_pickle('data/kp20k/test.pd')
body_text = testdf.body.tolist()
title_text = testdf.title.tolist()
seq2seq_Model_glove = load_model('data/kp20k/seq2seq_model_tutorial_glove.hdf5')
seq2seq_Model_fasttext = load_model('data/kp20k/seq2seq_model_tutorial_fasttext.hdf5')
seq2seq_Model_word2vec = load_model('data/kp20k/seq2seq_model_tutorial_word2vec.hdf5')
num_encoder_tokens, body_pp = load_text_processor('data/kp20k/body_pp.dpkl')
num_decoder_tokens, title_pp = load_text_processor('data/kp20k/title_pp.dpkl')


seq2seq_inf_glove = Seq2Seq_Inference(encoder_preprocessor=body_pp,
                                 decoder_preprocessor=title_pp,
                                 seq2seq_model=seq2seq_Model_glove)
# this method displays the predictions on random rows of the holdout set
#seq2seq_inf_glove.demo_model_predictions(n=10, issue_df=testdf)


bleu, rouge1_f, rouge1_p, rouge1_r, rouge2_f, rouge2_p, rouge2_r, rougel_f, rougel_p, rougel_r = \
    seq2seq_inf_glove.evaluate_model(body_text[:10000], title_text[:10000])
print("\n****** Glove BLEU scrore ******: %s" % str(bleu))
print("\n****** Glove ROUGE 1 f scrore ******: %s" % str(rouge1_f))
print("\n****** Glove ROUGE 1 precission scrore ******: %s" % str(rouge1_p))
print("\n****** Glove ROUGE 1 recall scrore ******: %s" % str(rouge1_r))
print("\n****** Glove ROUGE 2 f scrore ******: %s" % str(rouge2_f))
print("\n****** Glove ROUGE 2 precission scrore ******: %s" % str(rouge2_p))
print("\n****** Glove ROUGE 2 recall scrore ******: %s" % str(rouge2_r))
print("\n****** Glove ROUGE l f scrore ******: %s" % str(rougel_f))
print("\n****** Glove ROUGE l precission scrore ******: %s" % str(rougel_p))
print("\n****** Glove ROUGE l recall scrore ******: %s" % str(rougel_r))

seq2seq_inf_fasttext = Seq2Seq_Inference(encoder_preprocessor=body_pp,
                                 decoder_preprocessor=title_pp,
                                 seq2seq_model=seq2seq_Model_fasttext)
# this method displays the predictions on random rows of the holdout set
#seq2seq_inf_fasttext.demo_model_predictions(n=10, issue_df=testdf)


bleu, rouge1_f, rouge1_p, rouge1_r, rouge2_f, rouge2_p, rouge2_r, rougel_f, rougel_p, rougel_r = \
    seq2seq_inf_fasttext.evaluate_model(body_text[:10000], title_text[:10000])
print("\n****** Fasttext BLEU scrore ******: %s" % str(bleu))
print("\n****** Fasttext ROUGE 1 f scrore ******: %s" % str(rouge1_f))
print("\n****** Fasttext ROUGE 1 precission scrore ******: %s" % str(rouge1_p))
print("\n****** Fasttext ROUGE 1 recall scrore ******: %s" % str(rouge1_r))
print("\n****** Fasttext ROUGE 2 f scrore ******: %s" % str(rouge2_f))
print("\n****** Fasttext ROUGE 2 precission scrore ******: %s" % str(rouge2_p))
print("\n****** Fasttext ROUGE 2 recall scrore ******: %s" % str(rouge2_r))
print("\n****** Fasttext ROUGE l f scrore ******: %s" % str(rougel_f))
print("\n****** Fasttext ROUGE l precission scrore ******: %s" % str(rougel_p))
print("\n****** Fasttext ROUGE l recall scrore ******: %s" % str(rougel_r))

seq2seq_inf_word2vec = Seq2Seq_Inference(encoder_preprocessor=body_pp,
                                 decoder_preprocessor=title_pp,
                                 seq2seq_model=seq2seq_Model_word2vec)
# this method displays the predictions on random rows of the holdout set
#seq2seq_inf_word2vec.demo_model_predictions(n=10, issue_df=testdf)


bleu, rouge1_f, rouge1_p, rouge1_r, rouge2_f, rouge2_p, rouge2_r, rougel_f, rougel_p, rougel_r = \
    seq2seq_inf_word2vec.evaluate_model(body_text[:10000], title_text[:10000])
print("\n****** Word2vec BLEU scrore ******: %s" % str(bleu))
print("\n****** Word2vec ROUGE 1 f scrore ******: %s" % str(rouge1_f))
print("\n****** Word2vec ROUGE 1 precission scrore ******: %s" % str(rouge1_p))
print("\n****** Word2vec ROUGE 1 recall scrore ******: %s" % str(rouge1_r))
print("\n****** Word2vec ROUGE 2 f scrore ******: %s" % str(rouge2_f))
print("\n****** Word2vec ROUGE 2 precission scrore ******: %s" % str(rouge2_p))
print("\n****** Word2vec ROUGE 2 recall scrore ******: %s" % str(rouge2_r))
print("\n****** Word2vec ROUGE l f scrore ******: %s" % str(rougel_f))
print("\n****** Word2vec ROUGE l precission scrore ******: %s" % str(rougel_p))
print("\n****** Word2vec ROUGE l recall scrore ******: %s" % str(rougel_r))

#seq2seq_inf_custom = Seq2Seq_Inference(encoder_preprocessor=body_pp,
#                                 decoder_preprocessor=title_pp,
#                                 seq2seq_model=seq2seq_Model_custom)
# this method displays the predictions on random rows of the holdout set
#seq2seq_inf_custom.demo_model_predictions(n=10, issue_df=testdf)


#bleu = seq2seq_inf_custom.evaluate_model(body_text[:10000], title_text[:10000])
#print(f"\n****** Csutom BLEU scrore ******:\n {bleu}")
