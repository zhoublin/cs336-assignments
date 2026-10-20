import os
import regex as re
import copy
import time
import pickle
import psutil
import tracemalloc
import multiprocessing as mp
from typing import List, Iterable, Iterator
from collections import defaultdict

from .pretokenization_example import find_chunk_boundaries
from .plogger import ProjectLogger

class BPETrainer:
    def __init__(self,
        input_path: str,
        vocab_size: int,
        special_tokens: List[str],
        enable_parallel: bool = True
    ):
        self.train_data_path = input_path
        self.vocab_size = vocab_size
        if isinstance(special_tokens[0], str):
            self.special_tokens = [item.encode('utf-8') for item in special_tokens]
        else:
            self.special_tokens = special_tokens
        self.num_processes = os.cpu_count() if enable_parallel else 1
        self.pre_toke_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocab = [bytes([i]) for i in range(256)]
        self.merges = []

        self.logger = ProjectLogger("BPE").logger

    def get_vocab(self):
        total = self.vocab + self.special_tokens
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
            indices.append((0, pattern_indices[0].span()[0]))
            last_start = pattern_indices[0].span()[1]
        else:
            last_start = 0

        for obj in pattern_indices[1:]:
            # find the content without special token
            start, end = obj.span()
            assert start <= end, f"indices error: start={start}, end={end}"
            indices.append((last_start, start))
            last_start = end
        if len(chunk):
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
            boundaries = find_chunk_boundaries(f, self.num_processes, self.special_tokens)

        # parallize pre-tokenization
        proc_num = min(self.num_processes, len(boundaries))
        args = [[item1, item2, corpus_file] for (item1, item2) in zip(boundaries[:-1], boundaries[1:])]
        with mp.Pool(processes=proc_num) as pool:
            results = pool.map(
                self._worker, 
                args
            )

        # collect the pretokens
        self.pretokens_count = {}
        for item in results:
            # 不能用update，相同key覆盖
            for key in item:
                if key not in self.pretokens_count:
                    self.pretokens_count[key] = 0
                self.pretokens_count[key] += item[key]

        # prepare for learning
        self.pretokens_paircount = {}
        self.pretokens_list = {}
        for pretoken in self.pretokens_count:
            # split every bytes to a list
            pretoken_bytelist = [bytes([item]) for item in list(pretoken)]

            self.pretokens_paircount[pretoken] = {}
            self.pretokens_list[pretoken] = [bytes([item]) for item in list(pretoken)]
            for (token_front, token_tail) in zip(pretoken_bytelist[:-1], pretoken_bytelist[1:]):
                if (token_front, token_tail) not in self.pretokens_paircount[pretoken]:
                    self.pretokens_paircount[pretoken][(token_front, token_tail)] = 0
                self.pretokens_paircount[pretoken][(token_front, token_tail)] += self.pretokens_count[pretoken]

    def count_token_occur_in_text(self, merge_tokens):
            max_count, max_count_pair = 0, []
            count = defaultdict(int)
            for pretoken in self.pretokens_paircount:
                for pair in self.pretokens_paircount[pretoken]:
                    if pair in merge_tokens:
                        count[pair] += self.pretokens_paircount[pretoken][pair]
                        if count[pair] > max_count:
                            max_count = count[pair]
                            max_count_pair = [pair]
                        if count[pair] == max_count and pair not in max_count_pair:
                            max_count_pair.append(pair)

            return max_count_pair
    
    def train(self):
        # pre-tokenization
        start_time = time.perf_counter()
        self.preprocess_corpus(self.train_data_path)
        elapsed = time.perf_counter() - start_time
        self.logger.info(f"pre-tokenization elapsed: {elapsed:.6f}(s)")

        iteration = 1
        while len(self.vocab) + 1 < self.vocab_size:
            # self.logger.info(f"Iteration: {iteration}th")

            # 1. get the possible merged tokens
            merge_tokens = set()
            for pairtoken in self.pretokens_paircount:
                merge_tokens |= set(self.pretokens_paircount[pairtoken])

            # 2. get the frequency of potential merged tokens
            cand_pairs = self.count_token_occur_in_text(merge_tokens)
            if len(cand_pairs) == 0:
                break
            freq_pair = sorted(cand_pairs, reverse=True)[0]

            # 3. add into merges
            freq_token = (freq_pair[0] + freq_pair[1])
            if freq_pair in self.merges:
                raise Exception(f"Duplication in merges: {freq_pair}")
            self.merges.append(freq_pair)

            # 4. add into vocab
            if freq_token not in self.vocab:
                self.vocab.append(freq_token)

            records = {}
            for pretoken in self.pretokens_paircount:
                if freq_pair in self.pretokens_paircount[pretoken]:
                    records[pretoken] = (copy.deepcopy(self.pretokens_paircount[pretoken]), copy.deepcopy(self.pretokens_list[pretoken]))

            # 5. replace pretokens with new selected merged token
            self.update_token(freq_pair)
            
            iteration += 1

    def update_token(self, freq_pair):
        freq_token = freq_pair[0] + freq_pair[1]

        for pretoken in self.pretokens_list:
            if freq_pair in self.pretokens_paircount[pretoken]:

                current_tokn_list = self.pretokens_list[pretoken]

                index = current_tokn_list.index(freq_pair[0])
                while index < len(current_tokn_list):
                    if index + 1 < len(current_tokn_list) and \
                        tuple(current_tokn_list[index:index + 2]) == freq_pair:

                        # check pre-pair
                        if index > 0:
                            # reduce old
                            old_pair = (current_tokn_list[index - 1], current_tokn_list[index])
                            self.pretokens_paircount[pretoken][old_pair] -= self.pretokens_count[pretoken]

                            # increase new
                            new_pair = (current_tokn_list[index - 1], freq_token)
                            if new_pair not in self.pretokens_paircount[pretoken]:
                                self.pretokens_paircount[pretoken][new_pair] = 0
                            self.pretokens_paircount[pretoken][new_pair] += self.pretokens_count[pretoken]

                        # check tail-pair
                        if index + 2 < len(current_tokn_list):
                            # reduce old
                            old_pair = (current_tokn_list[index + 1], current_tokn_list[index + 2])
                            self.pretokens_paircount[pretoken][old_pair] -= self.pretokens_count[pretoken]

                            # increase new
                            new_pair = (freq_token, current_tokn_list[index + 2])
                            if new_pair not in self.pretokens_paircount[pretoken]:
                                self.pretokens_paircount[pretoken][new_pair] = 0
                            self.pretokens_paircount[pretoken][new_pair] += self.pretokens_count[pretoken]

                        self.pretokens_paircount[pretoken][tuple(current_tokn_list[index:index + 2])] -= self.pretokens_count[pretoken]
                        tmp = current_tokn_list[:index] + [freq_token]
                        if index + 2 < len(current_tokn_list):
                            tmp += current_tokn_list[index + 2:]
                        current_tokn_list = tmp
                        self.pretokens_list[pretoken] = current_tokn_list
                        continue

                    index += 1

                keys = set(self.pretokens_paircount[pretoken].keys())
                for item in keys:
                    if self.pretokens_paircount[pretoken][item] <= 0:
                        self.pretokens_paircount[pretoken].pop(item)

class Tokenizer:
    def __init__(self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.reverse_vocab = {vocab[_id]: _id for _id in vocab}
        self.merges = merges
        if special_tokens is not None and isinstance(special_tokens[0], str):
            self.special_tokens = [item.encode('utf-8') for item in special_tokens]
            # match the longer pattern first
            self.special_tokens = list(sorted(self.special_tokens, key=lambda x: len(x), reverse=True))
        else:
            self.special_tokens = special_tokens
        self.sp_pattern = rb"|".join([re.escape(item) for item in self.special_tokens]) if self.special_tokens else None
        self.pre_toke_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return Tokenizer(vocab, merges, special_tokens)

    def process_one_seq(self, text, pretokens_seq, pretokens_list):
        if text == b"":
            return
        
        # get pretokens and count the occurences
        for item in re.finditer(self.pre_toke_pattern, text.decode("utf-8", errors="replace")):
            pretoken = item.group().encode('utf-8')
            pretokens_seq.append(pretoken)
            pretokens_list[pretoken] = [bytes([item]) for item in list(pretoken)]

    def pretokenize(self, text):
        pattern_indices = list(re.finditer(self.sp_pattern, text)) if self.sp_pattern else []
        
        pretokens_seq = []
        pretokens_list = {}
        if len(pattern_indices) > 0:
            last_start = 0
            for obj in pattern_indices:
                # find the content without special token
                start, end = obj.span()
                if last_start < start:
                    self.process_one_seq(text[last_start:start], pretokens_seq, pretokens_list)
                pretokens_seq.append(obj.group())
                pretokens_list[obj.group()] = [obj.group()]

                last_start = end
        else:
            last_start = 0

        if len(text) and last_start < len(text):
            self.process_one_seq(text[last_start:len(text)], pretokens_seq, pretokens_list)

        return pretokens_seq, pretokens_list

    def encode(self, text: str) -> list[int]:
        if isinstance(text, str):
            byte_text = text.encode('utf-8')
        else:
            byte_text = text
        pretokens_seq, pretokens_list = self.pretokenize(byte_text)

        for pretoken in pretokens_list:
            token_seq = pretokens_list[pretoken]
            token_pairs = list(zip(token_seq[:-1], token_seq[1:]))

            cycle = 0
            while cycle < len(self.merges):
                merge = self.merges[cycle]
                matched = False
                if merge in token_pairs:
                    index1 = [1 if merge[0] == item else 0 for item in token_seq]
                    index2 = [1 if merge[1] == item else 0 for item in token_seq]
                    i = 0
                    while i < (len(index1) - 1):
                        if index1[i] == 1 and index2[i + 1] == 1:
                            token_seq = token_seq[:i] + [merge[0] + merge[1]] + token_seq[i + 2:]
                            index1 = [1 if merge[0] == item else 0 for item in token_seq]
                            index2 = [1 if merge[1] == item else 0 for item in token_seq]
                            token_pairs = list(zip(token_seq[:-1], token_seq[1:]))
                            matched = True
                        i += 1

                if matched:
                    cycle = 0
                else:
                    cycle += 1
            pretokens_list[pretoken] = token_seq

        token_id_seqs = {}
        for pretoken in pretokens_list:
            token_id_seqs[pretoken] = []
            for token in pretokens_list[pretoken]:
                token_id_seqs[pretoken].append(self.reverse_vocab[token])

        result = []
        for seq in pretokens_seq:
            result += token_id_seqs[seq]

        return result

    def get_boundary_by_pre_toke_pattern(self, text):
        last_start_point = 0
        for item in re.finditer(self.pre_toke_pattern, text.decode('utf-8')):
            last_start_point = item.span()[1]

        return last_start_point

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        chunk_size = 1024 * 100

        # Get total file size in bytes
        iterable.seek(0, os.SEEK_END)
        remain_file_size = iterable.tell()
        iterable.seek(0)
        chunk_size = min(chunk_size, remain_file_size)

        start_pos, end_pos = 0, chunk_size
        while True:
            iterable.seek(start_pos)
            mini_chunk = iterable.read(chunk_size).encode('utf-8')

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"" or remain_file_size <= 0:
                break

            # find the boundary
            pattern_indices = list(re.finditer(self.sp_pattern, mini_chunk)) if self.sp_pattern is not None else []
            if self.sp_pattern and len(pattern_indices):
                end_pos = pattern_indices[-1].span()[1]
            else:
                end_pos = self.get_boundary_by_pre_toke_pattern(mini_chunk)

            for item in self.encode(mini_chunk[:end_pos]):
                yield item

            start_pos = end_pos
            remain_file_size -= len(mini_chunk[:end_pos])

    def decode(self, ids: list[int]) -> str:
        output = b""
        for id in ids:
            output += self.vocab[id]
        return output.decode('utf-8', errors='replace')

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
    bpe_tokenizer = BPETrainer(input_path, vocab_size, special_tokens)
    bpe_tokenizer.train()
    return bpe_tokenizer.get_vocab(), bpe_tokenizer.get_merges()

if __name__ == '__main__':
    import time
    start_time = time.perf_counter()

    tracemalloc.start()
    try:
        vocab, merges = bpe_entry(
            "D:\\AI\\Learning\\CS336\\Assignment 1\\assignment1-basics\\data\\owt_train.txt",
            32000,
            ["<|endoftext|>"]
        )
        with open("openwebtext.vocab", 'wb') as f1, open("openwebtext.merges", 'wb') as f2:
            pickle.dump(vocab, f1)
            pickle.dump(merges, f2)
        current, peak = tracemalloc.get_traced_memory()
        print(f"当前跟踪内存: {current / 1024**2:.2f} MB")
        print(f"峰值跟踪内存: {peak / 1024**2:.2f} MB")
    finally:
        tracemalloc.stop()

    elapsed = time.perf_counter() - start_time
    print(f"elapsed: {elapsed:.6f}(s)")
