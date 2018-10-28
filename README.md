# Execute the seq2seq

## preprocessing

```
nohup python preprocessing_economics.py > logs/preprocessing_economics.log
```

```
nohup python preprocessing_recipes.py > logs/preprocessing_recipes.log
```

```
nohup python preprocessing_kp20k.py > logs/preprocessing_kp20k.log
```

## trainer

Full training:
```
nohup python3.6 trainer.py > /tmp/train_seq2seq.log 2>&1 &
```
Pretrained embeddings
```
nohup python3.6 trainer_myembeddings.py > /tmp/train_sigma_my_seq2seq.log 2>&1 &
```

For economics use 1400 epochs instead of 100 as there is only 7200 examples instead of 100k

Fasttext mebeddings
```
export CUDA_VISIBLE_DEVICES=2
nohup python trainer_pretrained_embeddings.py --n_epochs 100 --datasource recipes --emb_file /home/jmena/dev/data/fasttext/wiki.en.vec > logs/log_kp20k_fasttext.log &
```
Glove mebeddings
```
export CUDA_VISIBLE_DEVICES=3
nohup python trainer_pretrained_embeddings.py --n_epochs 100 --datasource recipes --emb_file /home/jmena/dev/data/glove/glove/glove.6B.300d.txt  > logs/log_kp20k_glove.log &
```
Word2vec mebeddings
```
export CUDA_VISIBLE_DEVICES=4
nohup python trainer_pretrained_embeddings.py --n_epochs 100 --datasource recipes --emb_file /home/jmena/dev/data/word2vec/GoogleNews-vectors-negative300.txt > logs/log_kp20k_word2vec.log &
```

## validation

```
nohup python validation_economics.py  > logs/validation_economics.log &
```
```
nohup python validation_recipes.py  > logs/validation_recipes.log &
```
```
nohup python validation_kp20k_12e.py  > logs/validation_kp20k_12e.log &
```
