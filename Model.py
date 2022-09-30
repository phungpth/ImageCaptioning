from keras import layers, models
from utils import data_generator

class ImageCaptioningModel:

    def __init__(self, max_length, vocab_size, embedding_dim, embedding_matrix, *args):
      self.__model = None
      self.my_args = args

      inputs1 = layers.Input(shape=(2048,))
      fe1 = layers.Dropout(0.5)(inputs1)
      fe2 = layers.Dense(256, activation='relu')(fe1)
      inputs2 = layers.Input(shape=(max_length,))
      se1 = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
      se2 = layers.Dropout(0.5)(se1)
      se3 = layers.LSTM(256)(se2)

      decoder1 = layers.merge.add([fe2, se3])
      decoder2 = layers.Dense(256, activation='relu')(decoder1)
      outputs = layers.Dense(vocab_size, activation='softmax')(decoder2)
      self.__model = models.Model(inputs=[inputs1, inputs2], outputs=outputs)
      self.__model.layers[2].set_weights([embedding_matrix])
      self.__model.layers[2].trainable = False
      self.__model.compile(loss='categorical_crossentropy', optimizer='adam')

    def train(self, epochs, steps):
      for i in range(epochs):
        generator = data_generator(*self.my_args)
        self.__model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

    def test(self, X, Y):
      return self.__model.evaluate(X,Y)
    
    def predict(self, X):
      return self.__model.predict(X)

    def save(self, weightName='/content/model_weights/model_30.h5'):    
      self.__model.save_weights(weightName)