from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import os


class InceptionV3Extractor:
    def __init__(self):
        super(InceptionV3Extractor, self).__init__()
        self.__model = None
        inception_v3 = InceptionV3(weights='imagenet')
        self.__model = Model(inception_v3.input, inception_v3.layers[-2].output)

    def extract(self, images):
        img = image.load_img(images, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        fea_vec = self.__model.predict(x)
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
        return fea_vec
    

class GloveExtractor:
    def __init__(self, vectorSize, vocab, glove_dir="glove.6B/glove.6B.200d.txt"):
        super().__init__()
        self.__model = None
        self.vectorSize = vectorSize
        self.vocab = vocab
        self.__embeddings_index = {}
        with open(glove_dir) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                self.__embeddings_index[word] = coefs

    def extract(self):
        embedding_dim = 200
        ixtoword = {}
        wordtoix = {}
        ix = 1
        for w in self.vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1
        vocab_size = len(ixtoword) + 1
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, i in wordtoix.items():
            embedding_vector = self.__embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix