# Keras Roberta(English) for Text Classification

### Requirements

see requirements.txt

### How to Convert Torch Roberta to TensorFlow Roberta:

0. Download `Fairseq` Roberta Pretrained Model: [https://github.com/pytorch/fairseq/blob/main/examples/roberta/README.md](https://github.com/pytorch/fairseq/blob/main/examples/roberta/README.md)

1. Convert roberta weights from PyTorch to Tensorflow

```
python convert_roberta_to_tf.py 
    --model_name your_model_name
    --cache_dir /path/to/pytorch/roberta 
    --tf_cache_dir /path/to/converted/roberta
```

2. Extract features as `tf_roberta_demo.py`

```
python tf_roberta_demo.py 
    --roberta_path /path/to/pytorch/roberta
    --tf_roberta_path /path/to/converted/roberta
    --tf_ckpt_name your_mode_name
```

### train text classification model on MRPC dataset

model training, see: `model_train.py`

model evaluate on test data, see `model_evaluate.py`

| params                                      | accuracy | F1  |
|---------------------------------------------|----------|-----|
| epoch:10, batch_size:16, max_seq_length:128 |||
| epoch:10, batch_size:16, max_seq_length:256 |||