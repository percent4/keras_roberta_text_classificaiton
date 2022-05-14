# -*- coding: utf-8 -*-
# @Time : 2022/5/14 11:41
# @Author : Jclian91
# @File : robeberta_tokernizer.py
# @Place : Minghang, Shanghai


# roberta tokenizer function for text pair
def tokenizer_encode(tokenizer, text1, text2, max_seq_length):
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
    pad_length = max_seq_length - len(token_ids)
    if pad_length >= 0:
        token_ids += [0] * pad_length
        segment_ids += [0] * pad_length
    else:
        token_ids = token_ids[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]

    return token_ids, segment_ids
