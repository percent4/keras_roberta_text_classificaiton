import os
import tensorflow as tf
from keras_roberta.roberta import build_bert_model
from keras_roberta.tokenizer import RobertaTokenizer
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
import numpy as np
import argparse


if __name__ == '__main__':
    roberta_path = 'roberta-base'
    tf_roberta_path = 'tf_roberta_base'
    tf_ckpt_name = 'tf_roberta_base.ckpt'
    vocab_path = 'keras_roberta'

    config_path = os.path.join(tf_roberta_path, 'bert_config.json')
    checkpoint_path = os.path.join(tf_roberta_path, tf_ckpt_name)
    if os.path.splitext(checkpoint_path)[-1] != '.ckpt':
        checkpoint_path += '.ckpt'

    gpt_bpe_vocab = os.path.join(vocab_path, 'encoder.json')
    gpt_bpe_merge = os.path.join(vocab_path, 'vocab.bpe')
    roberta_dict = os.path.join(roberta_path, 'dict.txt')

    tokenizer = RobertaTokenizer(gpt_bpe_vocab, gpt_bpe_merge, roberta_dict)
    model = build_bert_model(config_path, checkpoint_path, roberta=True)  # 建立模型，加载权重

    # 编码测试
    text1 = "hello, world!"
    text2 = "This is Roberta!"
    sep = [tokenizer.sep_token]
    cls = [tokenizer.cls_token]
    # 1. 先用'bpe_tokenize'将文本转换成bpe tokens
    tokens1 = cls + tokenizer.bpe_tokenize(text1) + sep
    tokens2 = sep + tokenizer.bpe_tokenize(text2) + sep
    # 2. 最后转换成id
    token_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
    token_ids2 = tokenizer.convert_tokens_to_ids(tokens2)
    token_ids = token_ids1 + token_ids2
    segment_ids = [0] * len(token_ids1) + [1] * len(token_ids2)
    print(token_ids)
    print(segment_ids)

    print('\n ===== tf model predicting =====\n')
    our_output = model.predict([np.array([token_ids]), np.array([segment_ids])])
    print(our_output)

    print('\n ===== torch model predicting =====\n')
    roberta = FairseqRobertaModel.from_pretrained(roberta_path)
    roberta.eval()  # disable dropout

    input_ids = roberta.encode(text1, text2).unsqueeze(0)  # batch of size 1
    print(input_ids)
    their_output = roberta.model(input_ids, features_only=True)[0]
    print(their_output)

    # print('\n ===== reloading and predicting =====\n')
    # model.save('test.model')
    # del model
    # model = keras.models.load_model('test.model')
    # print(model.predict([np.array([token_ids]), np.array([segment_ids])]))

