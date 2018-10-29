import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import logging
import numpy as np
import dill as dpickle
from tqdm import tqdm, tqdm_notebook
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from utils import extract_decoder_model, extract_encoder_model
from rouge import Rouge


class Seq2Seq_Inference(object):
    def __init__(self,
                 encoder_preprocessor,
                 decoder_preprocessor,
                 seq2seq_model):

        self.pp_body = encoder_preprocessor
        self.pp_title = decoder_preprocessor
        self.seq2seq_model = seq2seq_model
        self.encoder_model = extract_encoder_model(seq2seq_model)
        self.decoder_model = extract_decoder_model(seq2seq_model)
        self.default_max_len_title = self.pp_title.padding_maxlen
        self.nn = None
        self.rec_df = None
        self.rouge = Rouge()

    def generate_issue_title(self,
                             raw_input_text,
                             max_len_title=None):
        """
        Use the seq2seq model to generate a title given the body of an issue.

        Inputs
        ------
        raw_input: str
            The body of the issue text as an input string

        max_len_title: int (optional)
            The maximum length of the title the model will generate

        """
        if max_len_title is None:
            max_len_title = self.default_max_len_title
        # get the encoder's features for the decoder
        raw_tokenized = self.pp_body.transform([raw_input_text])
        body_encoding = self.encoder_model.predict(raw_tokenized)
        # we want to save the encoder's embedding before its updated by decoder
        #   because we can use that as an embedding for other tasks.
        original_body_encoding = body_encoding
        state_value = np.array(self.pp_title.token2id['_start_']).reshape(1, 1)

        decoded_sentence = []
        stop_condition = False
        while not stop_condition:
            preds, st = self.decoder_model.predict([state_value, body_encoding])

            # We are going to ignore indices 0 (padding) and indices 1 (unknown)
            # Argmax will return the integer index corresponding to the
            #  prediction + 2 b/c we chopped off first two
            pred_idx = np.argmax(preds[:, :, 2:]) + 2

            # retrieve word from index prediction
            pred_word_str = self.pp_title.id2token[pred_idx]

            if pred_word_str == '_end_' or len(decoded_sentence) >= max_len_title:
                stop_condition = True
                break
            decoded_sentence.append(pred_word_str)

            # update the decoder for the next word
            body_encoding = st
            state_value = np.array(pred_idx).reshape(1, 1)

        return original_body_encoding, ' '.join(decoded_sentence)


    def print_example(self,
                      i,
                      body_text,
                      title_text,
                      url,
                      threshold):
        """
        Prints an example of the model's prediction for manual inspection.
        """
        if i:
            print('\n\n==============================================')
            print('============== Example # %s =================\n' % str(i))

        if url:
            print(url)

        print("Issue Body:\n %s \n" % body_text)

        if title_text:
            print("Original Title:\n %s" % title_text)

        emb, gen_title = self.generate_issue_title(body_text)
        print("\n****** Machine Generated Title (Prediction) ******:\n %s" % gen_title)

        if self.nn:
            # return neighbors and distances
            n, d = self.nn.get_nns_by_vector(emb.flatten(), n=4,
                                             include_distances=True)
            neighbors = n[1:]
            dist = d[1:]

            if min(dist) <= threshold:
                cols = ['issue_url', 'title', 'body']
                dfcopy = self.rec_df.iloc[neighbors][cols].copy(deep=True)
                dfcopy['dist'] = dist
                similar_issues_df = dfcopy.query('dist <= %s' % str(threshold))

                print("\n**** Similar Issues (using encoder embedding) ****:\n")
                #display(similar_issues_df)


    def demo_model_predictions(self,
                               n,
                               issue_df,
                               threshold=1):
        """
        Pick n random Issues and display predictions.

        Input:
        ------
        n : int
            Number of issues to display from issue_df
        issue_df : pandas DataFrame
            DataFrame that contains two columns: `body` and `issue_title`.
        threshold : float
            distance threshold for recommendation of similar issues.

        Returns:
        --------
        None
            Prints the original issue body and the model's prediction.
        """
        # Extract body and title from DF
        body_text = issue_df.body.tolist()
        title_text = issue_df.title.tolist()
        url = ''#issue_df.issue_url.tolist()

        demo_list = np.random.randint(low=1, high=len(body_text), size=n)
        for i in demo_list:
            self.print_example(i,
                               body_text=body_text[i],
                               title_text=title_text[i],
                               url='',
                               threshold=threshold)


    def evaluate_model(self, holdout_bodies, holdout_titles):
        """
        Method for calculating BLEU Score.

        Parameters
        ----------
        holdout_bodies : List[str]
            These are the issue bodies that we want to summarize
        holdout_titles : List[str]
            This is the ground truth we are trying to predict --> issue titles

        Returns
        -------
        bleu : float
            The BLEU Score

        """
        bleus = list()
        rouge1_fs, rouge1_ps, rouge1_rs = list(), list(), list()
        rouge2_fs, rouge2_ps, rouge2_rs = list(), list(), list()
        rougel_fs, rougel_ps, rougel_rs = list(), list(), list()

        assert len(holdout_bodies) == len(holdout_titles)
        num_examples = len(holdout_bodies)

        logging.warning('Generating predictions.')
        # step over the whole set TODO: parallelize this
        for i in tqdm(range(num_examples)):
            _, yhat = self.generate_issue_title(holdout_bodies[i])
            current_actual = self.pp_title.process_text([holdout_titles[i]])[0]
            #print(current_actual)
            current_predicted = self.pp_title.process_text([yhat])[0]
            #print(current_predicted)
            bleu_score = sentence_bleu([current_actual], current_predicted, weights=(1, 0, 0, 0))
            bleus.append(bleu_score)
            #print(bleu_score)
            rouge_scores = self.rouge.get_scores(current_predicted, current_actual)
            rouge1_fs.append(rouge_scores[0]['rouge-1']['f'])
            rouge1_ps.append(rouge_scores[0]['rouge-1']['p'])
            rouge1_rs.append(rouge_scores[0]['rouge-1']['r'])
            rouge2_fs.append(rouge_scores[0]['rouge-2']['f'])
            rouge2_ps.append(rouge_scores[0]['rouge-2']['p'])
            rouge2_rs.append(rouge_scores[0]['rouge-2']['r'])
            rougel_fs.append(rouge_scores[0]['rouge-l']['f'])
            rougel_ps.append(rouge_scores[0]['rouge-l']['p'])
            rougel_rs.append(rouge_scores[0]['rouge-l']['r'])
        # calculate BLEU score
        logging.warning('Calculating BLEU.')
        bleus_array = np.array(bleus, dtype=float)
        rouge1_f_array = np.array(rouge1_fs, dtype=float)
        rouge1_p_array = np.array(rouge1_ps, dtype=float)
        rouge1_r_array = np.array(rouge1_rs, dtype=float)
        rouge2_f_array = np.array(rouge2_fs, dtype=float)
        rouge2_p_array = np.array(rouge2_ps, dtype=float)
        rouge2_r_array = np.array(rouge2_rs, dtype=float)
        rougel_f_array = np.array(rougel_fs, dtype=float)
        rougel_p_array = np.array(rougel_ps, dtype=float)
        rougel_r_array = np.array(rougel_rs, dtype=float)
        bleu = np.average(bleus_array)
        rouge1_f = np.average(rouge1_f_array)
        rouge1_p = np.average(rouge1_p_array)
        rouge1_r = np.average(rouge1_r_array)
        rouge2_f = np.average(rouge2_f_array)
        rouge2_p = np.average(rouge2_p_array)
        rouge2_r = np.average(rouge2_r_array)
        rougel_f = np.average(rougel_f_array)
        rougel_p = np.average(rougel_p_array)
        rougel_r = np.average(rougel_r_array)
        return bleu, rouge1_f, rouge1_p, rouge1_r, rouge2_f, rouge2_p, rouge2_r, rougel_f, rougel_p, rougel_r
