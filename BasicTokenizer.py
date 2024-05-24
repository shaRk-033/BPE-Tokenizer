class BasicTokenizer():
    
    def __init__(self, vocab_size):
        
        self.vocab = {idx : bytes([idx]) for idx in range(256)}
        self.bigram_tree = {}
        self.vocab_size = vocab_size
    
    def freq(self, tokens):
        
        stats = {}
        for id1, id2 in zip(tokens, tokens[1:]):
            stats[(id1, id2)]  = stats.get((id1, id2), 0) + 1
        return stats
    
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
    
    def train(self,txt):
        req = self.vocab_size - 256
        tokens = list((txt.encode('utf-8')))
        for i in range(req):
            mapz = self.freq(tokens)
            maxi = max(mapz, key = mapz.get)
            tokens = self.replace(tokens, maxi, 256 + i)
            self.bigram_tree[maxi] = 256 + i
        
        for (p0,p1),ix in self.bigram_tree.items():
            self.vocab[ix] = self.vocab[p0] + self.vocab[p1]
            
    def encode(self, txt):
        
        tokenz = list(map(int, txt.encode('utf-8')))
        while True:
            stats = self.freq(tokenz)
            maxi = min(stats, key = lambda p: self.bigram_tree.get(p, float('inf')))
            if maxi not in self.bigram_tree:
                break
            tokenz = self.replace(tokenz, maxi, self.bigram_tree[maxi])
        return tokenz

    def decode(self, tokens):
        
        t = b"".join(self.vocab[idx] for idx in tokens)
        txt = t.decode('utf-8', errors = 'replace')
        return txt
