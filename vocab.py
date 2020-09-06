
class Vocabulary:
    def __init__(self):
        self.index2item = []
        self.item2index = {}

    def __len__(self):
        return len(self.item2index)

    def __contains__(self, item):
        return item in self.item2index.keys()

    def add_item(self, item):
        index = len(self.item2index)
        self.index2item.append(item)
        self.item2index[item] = index

    def get_item(self, index):
        return self.index2item[index]

    def get_index(self, item):
        return self.item2index[item]

    def save(self, vocab_file):
        with open(vocab_file, 'w') as f:
            for word in self.item2index:
                print(word, file=f)

    @classmethod
    def load(cls, vocab_file):
        vocab = cls()
        with open(vocab_file) as f:
            for line in f:
                word = line.strip()
                vocab.item2index[word] = len(vocab.item2index)
                vocab.index2item.append(word)
        return vocab
