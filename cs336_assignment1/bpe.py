import os
import regex as re
import multiprocessing as mp
import argparse
from collections import Counter, defaultdict

from pretokenization_example import find_chunk_boundaries

class BytePairEncoder:
    def __init__(self, sep_token, enable_parallel):
        self.sep_tokens = sep_token
        self.num_processes = os.cpu_count() if enable_parallel else 1
        self.pre_toke_pattern = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocab = [chr(byte).encode('utf-8') for byte in range(256)]

    def get_vocab(self):
        return self.sep_tokens + self.vocab

    def count_token_occur_in_text(self, merge_tokens):
        count = defaultdict(int)
        for token in merge_tokens:
            for key in self.pair_count:
                for pair in self.pair_count[key]:
                    mtoken = (pair[0] + pair[1])
                    if token == mtoken:
                        count[token] += self.pair_count[key][pair]
        return count

    def update_token(self, freq_token):
        for key in self.pair_count:
            if freq_token in key:
                # merge token
                tmp = self.pair_count[key].copy()
                for pair in self.pair_count[key]:
                    mtoken = pair[0] + pair[1]

                    if freq_token == mtoken:
                        inc_prefix = defaultdict(int)
                        inc_suffix = defaultdict(int)
                        # remove pair
                        cnt = tmp.pop(pair)
                        # reduce
                        for t_pair in self.pair_count[key]:
                            # left pair
                            if t_pair[1] == pair[0] and (t_pair[0] + mtoken) in key:
                                tmp[t_pair] -= cnt
                                inc_prefix[t_pair[0]] += cnt
                                if tmp[t_pair] == 0:
                                    tmp.pop(t_pair)
                            # right pair
                            if t_pair[0] == pair[1] and (mtoken + t_pair[1]) in key:
                                tmp[t_pair] -= cnt
                                inc_suffix[t_pair[1]] += cnt
                                if tmp[t_pair] == 0:
                                    tmp.pop(t_pair)
                        # increase
                        for pref in inc_prefix:
                            tmp[(pref, mtoken)] += inc_prefix[pref]
                        for suff in inc_suffix:
                            tmp[(mtoken, suff)] += inc_suffix[suff]
                        break

                self.pair_count[key] = tmp

    def _worker(self, args):
        st, ed, corpus_file = args
        with open(corpus_file, 'rb') as f:
            f.seek(st)
            chunk = f.read(ed - st)
        sub_corpus = chunk.split(self.sep_tokens)
        pretokens_count = defaultdict(int)
        for cp in sub_corpus:
            if cp == b"":
                continue
            for item in re.finditer(self.pre_toke_pattern, cp):
                pretokens_count[item.group()] += 1
        print(f" --- preprocess subproc: {os.getpid()} done.")
        return Counter(pretokens_count)


    def preprocess_corpus(self, corpus_file):
        with open(corpus_file, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, self.sep_tokens)

        proc_num = min(self.num_processes, len(boundaries))
        args = [[item1, item2, corpus_file] for (item1, item2) in zip(boundaries[:-1], boundaries[1:])]
        with mp.Pool(processes=proc_num) as pool:
            results = pool.map(
                self._worker, 
                args
            )

        self.pretokens_count = sum(results, Counter())
        self.pretokens_list = {key: [chr(item).encode('utf-8') for item in list(key)] for key in self.pretokens_count}
        self.pair_count = {}
        for key in self.pretokens_list:
            pcnt = defaultdict(int)
            iter_pretoken = self.pretokens_list[key]
            for (token_front, token_tail) in zip(iter_pretoken[:-1], iter_pretoken[1:]):
                pcnt[(token_front, token_tail)] += self.pretokens_count[key]
            self.pair_count[key] = pcnt

    def __call__(self, corpus_file):
        self.preprocess_corpus(corpus_file)

        next_vocab = self.vocab.copy()
        iter = 1
        while True:
            print(f"Iter: {iter}")
            merge_tokens = set()
            for pairtoken in self.pair_count:
                for (token_front, token_tail) in self.pair_count[pairtoken]:
                    merge_tokens.add(token_front + token_tail)

            token_counts = self.count_token_occur_in_text(merge_tokens) # N^4
            if sum(list(token_counts.values())) == 0:
                break

            sorted_ext = sorted(token_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
            freq_token = sorted_ext[0][0]
            if freq_token not in next_vocab:
                next_vocab.append(freq_token)
                print(f"=== Add new token: {freq_token}, freq: {sorted_ext[0][1]}")

            # replace pretokens with new selected merged token
            self.update_token(freq_token)

            print(f"=== vocab size, old: {len(self.vocab)}, new: {len(next_vocab)}")
            if len(self.vocab) < len(next_vocab):
                self.vocab = next_vocab

            iter += 1

def parse_arguments():
    parser = argparse.ArgumentParser("Byte-Pair Encoding.")
    parser.add_argument(
        "-s",
        "--special_token_file",
        type=str,
        # required=True,
        help="Specifiy the file contains special tokens one item per line."
    )

    parser.add_argument(
        "-p",
        "--enable_parallel",
        action="store_true",
        help="Enable parallelizing pre-tokenization stage."
    )

    parser.add_argument(
            "-c",
            "--corpus_file",
            type=str,
            required=True,
            help="Specifiy the corpus file."
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    bpe = BytePairEncoder(b"<|endoftext|>", args.enable_parallel)
    bpe(args.corpus_file)
    print(bpe.get_vocab())
    with open("vocab.txt", 'wb') as f:
        for i, item in enumerate(bpe.get_vocab()):
            f.write(f"{i},{item}\n")