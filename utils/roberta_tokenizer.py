# -*- coding: utf-8 -*-
# @Time : 2022/5/14 11:41
# @Author : Jclian91
# @File : roberta_tokenizer.py
# @Place : Minghang, Shanghai


# roberta tokenizer function for text pair
def tokenizer_encode(tokenizer, text, max_seq_length):
    sep = [tokenizer.sep_token]
    cls = [tokenizer.cls_token]
    # 1. 先用'bpe_tokenize'将文本转换成bpe tokens
    tokens1 = cls + tokenizer.bpe_tokenize(text) + sep
    # 2. 最后转换成id
    token_ids = tokenizer.convert_tokens_to_ids(tokens1)
    segment_ids = [0] * len(token_ids)
    pad_length = max_seq_length - len(token_ids)
    if pad_length >= 0:
        token_ids += [0] * pad_length
        segment_ids += [0] * pad_length
    else:
        token_ids = token_ids[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]

    return token_ids, segment_ids
