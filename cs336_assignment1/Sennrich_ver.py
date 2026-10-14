from collections import Counter, defaultdict

class BytePairEncoder:
    def __init__(self, sep_tokens):
        self.sep_tokens = sep_tokens
        self.vocab = self.sep_tokens + [chr(byte).encode('utf-8') for byte in range(256)]
        print("=== original vocab:", self.vocab)

    def pre_tokenize(self, input_t):
        pretokens = [seq.encode('utf-8') for seq in input_t.replace('\n', " ").split(" ")]
        self.pretokens_count = Counter(pretokens)
        self.pretokens_list = {key: [chr(item).encode('utf-8') for item in list(key)] for key in pretokens}
        self.pair_count = {}
        for key in self.pretokens_list:
            pcnt = defaultdict(int)
            iter_pretoken = self.pretokens_list[key]
            for (token_front, token_tail) in zip(iter_pretoken[:-1], iter_pretoken[1:]):
                pcnt[(token_front, token_tail)] += self.pretokens_count[key]
            self.pair_count[key] = pcnt
        print("=== pre_tokenize:", self.pretokens_count)
        print("=== pre_tokenize list:", self.pretokens_list)
        print("=== pre_tokenize pair:", self.pair_count)

    def count_token_occur_in_text(self, merge_tokens):
        count = defaultdict(int)
        for token in merge_tokens:
            for key in self.pair_count:
                for pair in self.pair_count[key]:
                    mtoken = (pair[0] + pair[1])
                    if token == mtoken:
                        count[token] += self.pair_count[key][pair]
                        print(f"  === Token {token} in pair:{pair}, count: {count}")
        return count

    def update_token(self, freq_token):
        for key in self.pair_count:
            print("debug:", freq_token, key)
            if freq_token in key:
                # merge token
                tmp = self.pair_count[key].copy()
                for pair in self.pair_count[key]:
                    mtoken = pair[0] + pair[1]
                    print("  debug:", mtoken, freq_token)
                    if freq_token == mtoken:
                        print(f" --- pretokens_list updated, freq_token: {freq_token}")
                        print(f" --- pre:")
                        print(tmp)

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

                        

                        print(f" --- after:")
                        print(tmp)
                        break
                self.pair_count[key] = tmp

    def __call__(self, input_t):
        self.pre_tokenize(input_t)  # M pretokens

        next_vocab = self.vocab.copy()
        iter = 1
        while True:
            print(f"Iter: {iter}")
            merge_tokens = set()
            for pairtoken in self.pair_count:
                print(f" --- pretoken: {pairtoken}")
                for (token_front, token_tail) in self.pair_count[pairtoken]:
                    print(" --- Get new token:", token_front, token_tail)
                    merge_tokens.add(token_front + token_tail)
            print(f"=== merge_tokens: {merge_tokens}")

            token_counts = self.count_token_occur_in_text(merge_tokens) # N^4
            if sum(list(token_counts.values())) == 0:
                break

            print(f"=== token_counts: {token_counts}")
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

if __name__ == '__main__':
    input_text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

    bpe = BytePairEncoder(['<|endoftext|>'])
    bpe(input_text)
    print(bpe.vocab)