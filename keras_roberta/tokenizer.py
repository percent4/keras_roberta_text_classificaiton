import os
import json
import numpy as np
from keras_roberta.gpt2_bpe import Encoder


def convert_by_vocab(vocab, items, unk_idx=3):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        if item in vocab:
            output.append(vocab[item])
        else:
            output.append(unk_idx)
    return output


class RobertaTokenizer:
    def __init__(self, gpt_vocab_file, gpt_merge_file, roberta_vocab_file):
        """
        :param gpt_vocab_file: gpt2_bpe vocab file (DEFAULT_ENCODER_JSON in 'gpt2_bpe.py')
        :param gpt_merge_file:  gpt2_bpe merge file (DEFAULT_VOCAB_BPE in 'gpt2_bpe.py')
        :param roberta_vocab_file: 'dict.txt' in downloaded pre-trained keras_roberta model file
        """
        with open(gpt_vocab_file, 'r') as f:
            gpt_vocab = json.load(f)
        with open(gpt_merge_file, 'r', encoding="utf-8") as f:
            gpt_bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in gpt_bpe_data.split('\n')[1:-1]]
        gpt_vocab_rev = {v: k for k, v in gpt_vocab.items()}
        self.oldid2newid = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        self.newid2oldid = {}  # No bos_token, eos_token,...

        with open(roberta_vocab_file, 'r') as f:
            for i, line in enumerate(f):
                new_id = i + 4
                old_id, count = line.strip().split()
                if old_id.isnumeric():
                    self.oldid2newid[int(old_id)] = new_id
                    self.newid2oldid[new_id] = int(old_id)

        self.bpe = Encoder(encoder=gpt_vocab, bpe_merges=bpe_merges)
        self.bos_token = self.cls_token = "<s>"
        self.bos_idx = self.cls_idx = 0
        self.eos_token = self.sep_token = "</s>"
        self.eos_idx = self.sep_idx = 2
        self.unk_token = "<unk>"
        self.unk_idx = 3
        self.pad_token = '<pad>'
        self.pad_idx = 1
        self.mask_token = '<mask>'

    def bpe_tokenize(self, text):
        return self.bpe.encode(text)

    def detokenize(self, token_ids):
        if token_ids[0] == self.cls_idx:
            token_ids = token_ids[1:]  # remove <s>
        sentences = []
        sent = []
        for tid in token_ids:
            if tid != self.sep_idx:
                sent.append(tid)
            else:
                sentences.append(sent)
                sent = []
        sentences = [self.bpe.decode(self.convert_new_ids_to_old_ids(s)) for s in sentences]
        return sentences

    def convert_tokens_to_ids(self, tokens):
        new_token_ids = convert_by_vocab(self.oldid2newid, tokens, unk_idx=self.unk_idx)
        return new_token_ids

    def convert_new_ids_to_old_ids(self, ids):
        tokens = convert_by_vocab(self.newid2oldid, ids)
        return tokens


if __name__ == '__main__':
    tokenizer = RobertaTokenizer('encoder.json', 'vocab.bpe', 'dict.txt')
    text = "你好我是中文"
    sep = [tokenizer.sep_token]
    cls = [tokenizer.cls_token]
    # 1. 先用'bpe_tokenize'将文本转换成bpe tokens
    tokens = tokenizer.bpe_tokenize(text)
    # 2. 然后自行添加一些标志token
    tokens = cls + tokens + sep + tokens + sep
    print(tokens)
    # 3. 最后转换成id
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(token_ids)

    # 解码
    sentences = tokenizer.detokenize(token_ids)
    print(sentences)