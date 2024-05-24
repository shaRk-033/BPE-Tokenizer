import regex as re

class GPTTokenizer():
    def __init__(self, vocab_size):
        self.GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.vocab_size = vocab_size
        self.bigram_tree = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
    
    def split_text(self, text):
        return re.findall(self.GPT4_SPLIT_PATTERN, text)
    
    def freq(self, tokens, stats):
        for id1, id2 in zip(tokens, tokens[1:]):
            stats[(id1, id2)] = stats.get((id1, id2), 0) + 1
    
    def replace(self, tokens, pair, idx):
        newids = []
        i = 0
        while i < len(tokens):
            if tokens[i] == pair[0] and i < len(tokens) - 1 and tokens[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(tokens[i])
                i += 1
        return newids 
    
    def train(self, text):
        chunks = re.findall(self.GPT4_SPLIT_PATTERN, text)
        tokens = [list(chunk.encode('utf-8')) for chunk in chunks]
        
        for i in range(self.voacb_size - 256):
            stats = {}
            for token in tokens:
                self.freq(token, stats)
            maxi = max(stats, key=stats.get)
            tokens = [self.replace(token, maxi, 256 + i) for token in tokens]
            self.bigram_tree[maxi] = 256 + i
            self.vocab[256 + i] = self.vocab[maxi[0]] + self.vocab[maxi[1]]
            
    def _encode_chunk(self, tokens):
        while len(tokens)>=2:
            stats = {}
            self.freq(tokens, stats)
            pair = min(stats, key = lambda p: self.bigram_tree.get(p, float("inf")))
            if pair not in self.bigram_tree:
                break
            tokens = self.replace(tokens, pair, self.bigram_tree[pair])
        return tokens
    
    def encode(self, text):
        chunks = re.findall(self.GPT4_SPLIT_PATTERN, text)
        tokens = []
        for chunk in chunks:
            tks = list(chunk.encode('utf-8'))
            tks = self.encode_chunk(tks)
            tokens.extend(tks)
            
        return tokens
    
    def decode(self, tokens):
        x = (b"".join(self.vocab[idx] for idx in tokens))
        text = x.decode('utf-8')
        return text
