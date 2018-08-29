from keras.models import load_model
from sklearn.model_selection import train_test_split
from utils import load_text_processor
from utils_recipes import Seq2Seq_Inference
import pandas as pd

#read in data sample 2M rows (for speed of tutorial)
traindf, testdf = train_test_split(pd.read_csv('data/kp20k/kp20k.csv'),
                                   test_size=.10)
seq2seq_Model = load_model('data/kp20k/seq2seq_model_tutorial.hdf5')
num_encoder_tokens, body_pp = load_text_processor('data/kp20k/body_pp.dpkl')
num_decoder_tokens, title_pp = load_text_processor('data/kp20k/title_pp.dpkl')
seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=body_pp,
                                 decoder_preprocessor=title_pp,
                                 seq2seq_model=seq2seq_Model)
# this method displays the predictions on random rows of the holdout set
seq2seq_inf.demo_model_predictions(n=10, issue_df=testdf)

body_text = testdf.body.tolist()
title_text = testdf.title.tolist()
bleu = seq2seq_inf.evaluate_model(body_text[:1000], title_text[:1000])
print(f"\n****** BLEU scrore ******:\n {bleu}")
