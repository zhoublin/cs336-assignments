import os
import regex as re
import copy
import time
import pickle
from datetime import datetime
import multiprocessing as mp
import argparse
from typing import List, Tuple
from collections import Counter, defaultdict

from pretokenization_example import find_chunk_boundaries
from plogger import ProjectLogger

class BPETokenizer:
    def __init__(self,
        input_path: str,
        vocab_size: int,
        special_tokens: List[str],
        enable_parallel: bool = True
    ):
        self.train_data_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processes = os.cpu_count() if enable_parallel else 1
        self.pre_toke_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocab = [bytes([i]) for i in range(256)]
        self.merges = []

        self.logger = ProjectLogger("BPE").logger

    def get_vocab(self):
        total = self.special_tokens + self.vocab
        return {i: item for i, item in enumerate(total)}

    def get_merges(self):
        return self.merges

    def _worker(self, args):
        st, ed, corpus_file = args
        if st >= ed:
            return {}

        # read file content by index
        with open(corpus_file, 'rb') as f:
            f.seek(st)
            chunk = f.read(ed - st)

        # find the indices of special tokens
        pretokens_count = defaultdict(int)
        pattern = rb"|".join([re.escape(item) for item in self.special_tokens])
        pattern_indices = list(re.finditer(pattern, chunk))

        indices = []
        if len(pattern_indices) > 0:
            last_start = pattern_indices[0].span()[1]
        else:
            last_start = 0
        for obj in pattern_indices[1:]:
            # find the content without special token
            start, end = obj.span()
            assert start <= end, f"indices error: start={start}, end={end}"
            indices.append((last_start, start))
            last_start = end
        indices.append((last_start, len(chunk)))

        for strat, end in indices:
            cp = chunk[strat:end]
            if cp == b"":
                continue
            
            # get pretokens and count the occurences
            for item in re.finditer(self.pre_toke_pattern, cp.decode("utf-8", errors="replace")):
                pretokens_count[item.group().encode('utf-8')] += 1

        return pretokens_count

    def preprocess_corpus(self, corpus_file):
        # find the boundaries of every chunk
        with open(corpus_file, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, self.special_tokens[0])
        self.logger.info(f"#boundary: {len(boundaries)}: {boundaries}")

        # parallize pre-tokenization
        proc_num = min(self.num_processes, len(boundaries))
        args = [[item1, item2, corpus_file] for (item1, item2) in zip(boundaries[:-1], boundaries[1:])]
        with mp.Pool(processes=proc_num) as pool:
            results = pool.map(
                self._worker, 
                args
            )

        self.logger.info(results)
        # collect the pretokens
        self.pretokens_count = {}
        for item in results:
            self.pretokens_count.update(item)

        self.logger.info(f"pretokens_count: {self.pretokens_count}")

        # prepare for learning
        self.pretokens_paircount = {}
        self.pretokens_list = {}
        for pretoken in self.pretokens_count:
            # split every bytes to a list
            pretoken_bytelist = [bytes([item]) for item in list(pretoken)]

            self.pretokens_paircount[pretoken] = {}
            self.pretokens_list[pretoken] = []
            for (token_front, token_tail) in zip(pretoken_bytelist[:-1], pretoken_bytelist[1:]):
                if (token_front, token_tail) not in self.pretokens_paircount[pretoken]:
                    self.pretokens_paircount[pretoken][(token_front, token_tail)] = 0
                self.pretokens_paircount[pretoken][(token_front, token_tail)] += self.pretokens_count[pretoken]
                self.pretokens_list[pretoken].append((token_front, token_tail))

        self.logger.info(f"pretokens_paircount: {self.pretokens_paircount}")
        self.logger.info(f"pretokens_list: {self.pretokens_list}")

    def count_token_occur_in_text(self, merge_tokens):
            count = defaultdict(int)
            for pretoken in self.pretokens_paircount:
                for pair in self.pretokens_paircount[pretoken]:
                    if pair in merge_tokens:
                        count[pair] += self.pretokens_paircount[pretoken][pair]
            return count
    
    def train(self):
        # pre-tokenization
        self.preprocess_corpus(self.train_data_path)

        iteration = 1
        while len(self.vocab) - 1 < self.vocab_size:
            self.logger.info(f"Iteration: {iteration}th")

            # 1. get the possible merged tokens
            merge_tokens = set()
            for pairtoken in self.pretokens_paircount:
                merge_tokens |= set(self.pretokens_paircount[pairtoken])

            # 2. get the frequency of potential merged tokens
            token_counts = self.count_token_occur_in_text(merge_tokens)
            if sum(list(token_counts.values())) == 0:
                break

            # 3. get the most occurence token
            sorted_ext = sorted(token_counts.items(), key=lambda x: (x[1], x[0][0] + x[0][1]), reverse=True)
            freq_pair = sorted_ext[0][0]
            freq_token = (freq_pair[0] + freq_pair[1])
            self.logger.info(f"selected token: {freq_pair} -> {freq_token}")
            if freq_pair in self.merges:
                raise Exception(f"Duplication in merges: {freq_pair}")
            self.merges.append(freq_pair)

            # 4. add into vocab
            if freq_token not in self.vocab:
                self.vocab.append(freq_token)
                self.logger.info(f"New token: {freq_token}, freq: {sorted_ext[0][1]}")

            # 5. replace pretokens with new selected merged token
            self.update_token(freq_pair)

            self.logger.info(f"Vocab size: {len(self.vocab)}\n")
            
            iteration += 1

    def update_token(self, freq_pair):
        freq_token = freq_pair[0] + freq_pair[1]

        for pretoken in self.pretokens_list:
            if freq_pair in self.pretokens_list[pretoken]:
                self.logger.info(f"pretoken: {pretoken}")
                self.logger.info(f"current list: {self.pretokens_list[pretoken]}")

                i = 0
                while i < len(self.pretokens_list[pretoken]):
                    curt_pair = self.pretokens_list[pretoken][i]

                    if freq_pair == curt_pair:
                        # check pre-pair
                        if i > 0:
                            pre_pair = self.pretokens_list[pretoken][i - 1]
                            assert pre_pair[1] == freq_pair[0]

                            # update pretokens_list
                            new_pre_pair = (pre_pair[0], freq_token)
                            self.pretokens_list[pretoken][i - 1] = new_pre_pair

                            # update pretokens_paircount
                            self.pretokens_paircount[pretoken][pre_pair] -= self.pretokens_count[pretoken]

                            if new_pre_pair not in self.pretokens_paircount[pretoken]:
                                self.pretokens_paircount[pretoken][new_pre_pair] = 0
                            self.pretokens_paircount[pretoken][new_pre_pair] += self.pretokens_count[pretoken]

                        # check tail-pair
                        if i + 1 < len(self.pretokens_list[pretoken]):
                            tail_pair = self.pretokens_list[pretoken][i + 1]
                            assert freq_pair[1] == tail_pair[0]

                            # update pretokens_list
                            new_tail_pair = (freq_token, tail_pair[1])
                            self.pretokens_list[pretoken][i + 1] = new_tail_pair

                            # update pretokens_paircount
                            self.pretokens_paircount[pretoken][tail_pair] -= self.pretokens_count[pretoken]

                            if new_tail_pair not in self.pretokens_paircount[pretoken]:
                                self.pretokens_paircount[pretoken][new_tail_pair] = 0
                            self.pretokens_paircount[pretoken][new_tail_pair] += self.pretokens_count[pretoken]

                        # remove current-pair
                        self.pretokens_list[pretoken] = self.pretokens_list[pretoken][:i] + \
                                                        self.pretokens_list[pretoken][i + 1:]
                        self.pretokens_paircount[pretoken][curt_pair] -= self.pretokens_count[pretoken]

                    i += 1

                keys = list(self.pretokens_paircount[pretoken].keys())
                for item in keys:
                    if self.pretokens_paircount[pretoken][item] <= 0:
                        self.pretokens_paircount[pretoken].pop(item)

def bpe_entry(input_path: str, vocab_size: int, special_tokens: List[str]):
    """
    Input
        - [input_path]: str Path to a text file with BPE tokenizer training data.
        - [vocab_size]: int A positive integer that defines the maximum final vocabulary size (including 
        the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        - [special_tokens]: list[str] A list of strings to add to the vocabulary. During training, treat 
        them as hard boundaries that prevent merges across their spans, but do not include them when 
        computing merge statistics.

    Output
        - [vocab]: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the 
        vocabulary) to bytes (token bytes).
        - [merges]: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list 
        item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with 
        <token2>. The merges should be ordered by order of creation.
    """
    date = datetime.now().strftime('%Y-%m-%d')
    bpe_tokenizer = BPETokenizer(input_path, vocab_size, special_tokens)
    bpe_tokenizer.train()
    with open(f"vocab-{date}.pkl", 'wb') as f:
        pickle.dump(bpe_tokenizer.get_vocab(), f)
    with open(f"merge-{date}.pkl", 'wb') as f:
        pickle.dump(bpe_tokenizer.get_merges(), f)
    return bpe_tokenizer.get_vocab(), bpe_tokenizer.get_merges()


if __name__ == '__main__':
    vocab, merges = bpe_entry("D:\\AI\\Learning\\CS336\\Assignment 1\\assignment1-basics\\data\\TinyStoriesV2-GPT4-valid.txt", 10000, [b"<|endoftext|>"])
    print(vocab)
    print(merges)
