import pickle
import os.path
import string
from utils import load_doc

class Tokenizer:
    def __init__(self,
        vocab_threshold,
        vocab_file='./vocab.pkl',
        start_word="<startseq>",
        end_word="<endseq>",
        unk_word="<unk>",
        annotations_file='../cocoapi/annotations/captions_train2014.json',
        vocab_from_file=False):
        
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary successfully loaded from vocab.pkl file!')
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)
        
    def build_vocab(self):
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):

        doc = load_doc(self.annotations_file)
        
        descriptions = dict()

        for line in doc.split('\n'):
            tokens = line.split()
            image_id, image_desc = tokens[0], tokens[1:]
            if image_id in doc:
                if image_id not in descriptions:
                    descriptions[image_id] = list()
                descriptions[image_id].append(image_desc)
        all_train_captions = []
        for key, val in descriptions.items():
            for cap in val:
                all_train_captions.append(cap)
        word_counts = {}
        nsents = 0
        for sent in all_train_captions:
            nsents += 1
            for w in sent.split(' '):
                word_counts[w] = word_counts.get(w, 0) + 1

        words = [w for w in word_counts if word_counts[w] >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
        