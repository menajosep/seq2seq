# Execute the seq2seq

## preprocessing

```
nohup python3.6 preprocessing_kp20k.py > /tmp/seq2seq.log 2>&1 &
```

input: /home/bigplay/uniko/tensorflow/data/kp20k/ke20k_training.json
bad_words_list: data/kp20k/bad_words_list.p

## trainer

Full training:
```
nohup python3.6 trainer.py > /tmp/train_seq2seq.log 2>&1 &
```
Pretrained embeddings
```
nohup python3.6 trainer_myembeddings.py > /tmp/train_sigma_my_seq2seq.log 2>&1 &
```
Glove mebeddings
```
nohup python3.6 trainer_glove.py > /tmp/train_sigma_my_seq2seq.log 2>&1 &
```

## validation

```
nohup python3.6 validation_kp20k.py > /tmp/val_seq2seq.log 2>&1 &
```
