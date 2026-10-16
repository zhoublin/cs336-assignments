import multiprocessing as mp
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import regex as re

from .plogger import ProjectLogger
from .pretokenization_example import find_chunk_boundaries


class BPETokenizer:
    def __init__(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: List[str],
        enable_parallel: bool = True,
    ):
        self.train_data_path = input_path
        self.vocab_size = vocab_size
        if isinstance(special_tokens[0], str):
            self.special_tokens = [item.encode("utf-8") for item in special_tokens]
        else:
            self.special_tokens = special_tokens
        self.num_processes = os.cpu_count() if enable_parallel else 1
        self.pre_toke_pattern = (
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.vocab = [bytes([i]) for i in range(256)]
        self.merges = []

        # Filled in by preprocess_corpus()
        self.pretokens_count: Dict[bytes, int] = {}
        self.pretokens_paircount: Dict[bytes, Dict[Tuple[bytes, bytes], int]] = {}
        self.pretokens_list: Dict[bytes, List[bytes]] = {}

        self.logger = ProjectLogger("BPE").logger

    def get_vocab(self):
        total = self.vocab + self.special_tokens
        return {i: item for i, item in enumerate(total)}

    def get_merges(self):
        return self.merges

    def _worker(self, args):
        start_pos, end_pos, corpus_file = args
        if start_pos >= end_pos:
            return {}

        # Read the chunk of the file by byte offsets
        with open(corpus_file, "rb") as f:
            f.seek(start_pos)
            chunk = f.read(end_pos - start_pos)

        # Find the special tokens in the chunk
        pretokens_count = defaultdict(int)
        pattern = rb"|".join([re.escape(item) for item in self.special_tokens])
        special_matches = list(re.finditer(pattern, chunk))

        # Compute the spans of text between special tokens
        spans = []
        if len(special_matches) > 0:
            spans.append((0, special_matches[0].span()[0]))
            last_start = special_matches[0].span()[1]
        else:
            last_start = 0

        for match in special_matches[1:]:
            start, end = match.span()
            assert start <= end, f"indices error: start={start}, end={end}"
            spans.append((last_start, start))
            last_start = end
        if len(chunk):
            spans.append((last_start, len(chunk)))

        for start, end in spans:
            text_bytes = chunk[start:end]
            if text_bytes == b"":
                continue

            # Pre-tokenize and count occurrences
            text = text_bytes.decode("utf-8", errors="replace")
            for item in re.finditer(self.pre_toke_pattern, text):
                pretokens_count[item.group().encode("utf-8")] += 1

        return pretokens_count

    def preprocess_corpus(self, corpus_file):
        # Find the boundaries of every chunk
        with open(corpus_file, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, self.special_tokens)

        # Pre-tokenize in parallel
        proc_num = min(self.num_processes, len(boundaries))
        args = [
            [start, end, corpus_file]
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        with mp.Pool(processes=proc_num) as pool:
            results = pool.map(self._worker, args)

        # Merge the counts (dict.update would overwrite counts of identical keys)
        self.pretokens_count = {}
        for result in results:
            for key in result:
                if key not in self.pretokens_count:
                    self.pretokens_count[key] = 0
                self.pretokens_count[key] += result[key]

        # Prepare the data structures for training
        self.pretokens_paircount = {}
        self.pretokens_list = {}
        for pretoken in self.pretokens_count:
            # Split every pretoken into a list of single bytes
            pretoken_bytelist = [bytes([item]) for item in list(pretoken)]

            self.pretokens_paircount[pretoken] = {}
            self.pretokens_list[pretoken] = [bytes([item]) for item in list(pretoken)]
            for token_front, token_tail in zip(pretoken_bytelist[:-1], pretoken_bytelist[1:]):
                pair = (token_front, token_tail)
                if pair not in self.pretokens_paircount[pretoken]:
                    self.pretokens_paircount[pretoken][pair] = 0
                self.pretokens_paircount[pretoken][pair] += self.pretokens_count[pretoken]

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
        # Pre-tokenization
        self.preprocess_corpus(self.train_data_path)

        while len(self.vocab) + 1 < self.vocab_size:
            # 1. Collect the candidate pairs to merge
            merge_tokens = set()
            for pretoken in self.pretokens_paircount:
                merge_tokens |= set(self.pretokens_paircount[pretoken])

            # 2. Find the most frequent pair(s) (ties broken by lexicographic order)
            cand_pairs = self.count_token_occur_in_text(merge_tokens)
            if len(cand_pairs) == 0:
                break
            freq_pair = sorted(cand_pairs, reverse=True)[0]

            # 3. Record the merge
            freq_token = freq_pair[0] + freq_pair[1]
            if freq_pair in self.merges:
                raise Exception(f"Duplication in merges: {freq_pair}")
            self.merges.append(freq_pair)

            # 4. Add the new token to the vocabulary
            if freq_token not in self.vocab:
                self.vocab.append(freq_token)

            # 5. Replace the pair in all pretokens with the new merged token
            self.update_token(freq_pair)

    def update_token(self, freq_pair):
        freq_token = freq_pair[0] + freq_pair[1]

        for pretoken in self.pretokens_list:
            if freq_pair not in self.pretokens_paircount[pretoken]:
                continue

            pair_count = self.pretokens_paircount[pretoken]
            occurrences = self.pretokens_count[pretoken]
            token_list = self.pretokens_list[pretoken]

            index = token_list.index(freq_pair[0])
            while index < len(token_list):
                if index + 1 < len(token_list) and tuple(token_list[index:index + 2]) == freq_pair:
                    # Update the pair with the preceding token
                    if index > 0:
                        old_pair = (token_list[index - 1], token_list[index])
                        pair_count[old_pair] -= occurrences

                        new_pair = (token_list[index - 1], freq_token)
                        if new_pair not in pair_count:
                            pair_count[new_pair] = 0
                        pair_count[new_pair] += occurrences

                    # Update the pair with the following token
                    if index + 2 < len(token_list):
                        old_pair = (token_list[index + 1], token_list[index + 2])
                        pair_count[old_pair] -= occurrences

                        new_pair = (freq_token, token_list[index + 2])
                        if new_pair not in pair_count:
                            pair_count[new_pair] = 0
                        pair_count[new_pair] += occurrences

                    # Remove the merged pair itself
                    pair_count[tuple(token_list[index:index + 2])] -= occurrences

                    # Replace the pair with the merged token
                    merged_list = token_list[:index] + [freq_token]
                    if index + 2 < len(token_list):
                        merged_list += token_list[index + 2:]
                    token_list = merged_list
                    self.pretokens_list[pretoken] = token_list
                    continue

                index += 1

            # Drop pairs whose count fell to zero
            for pair in set(pair_count.keys()):
                if pair_count[pair] <= 0:
                    pair_count.pop(pair)


def bpe_entry(input_path: str, vocab_size: int, special_tokens: List[str]):
    """
    Input
        - [input_path]: str Path to a text file with BPE tokenizer training data.
        - [vocab_size]: int A positive integer that defines the maximum final vocabulary size
        (including the initial byte vocabulary, vocabulary items produced from merging, and
        any special tokens).
        - [special_tokens]: list[str] A list of strings to add to the vocabulary. During
        training, treat them as hard boundaries that prevent merges across their spans, but
        do not include them when computing merge statistics.

    Output
        - [vocab]: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in
        the vocabulary) to bytes (token bytes).
        - [merges]: list[tuple[bytes, bytes]] A list of BPE merges produced from training.
        Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1>
        was merged with <token2>. The merges should be ordered by order of creation.
    """
    bpe_tokenizer = BPETokenizer(input_path, vocab_size, special_tokens)
    bpe_tokenizer.train()
    return bpe_tokenizer.get_vocab(), bpe_tokenizer.get_merges()
