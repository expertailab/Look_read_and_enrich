import random
import sys
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold, train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense, Concatenate, Add, InputLayer, Reshape,BatchNormalization, Dropout, Multiply, LSTM
from keras.models import Model, Sequential

class CaptionCorpus:
    def __init__(self):
        self.tokenizer = None
        self.list_captions = []
        self.captions_processed = []
        self.categories = []
        self.labels_processed = None
        self.max_len = 1000

    def generate_corpus(self):
        with open("./jsons/scigraph.json", "r", encoding="utf-8", errors="surrogatepass") as file:
            dataset = json.load(file)
        for doc in dataset:
            self.list_captions.extend(doc["captions"])
            self.categories.append(doc["category"])
    def process_corpus(self): 
        with open("./saved/tokenizers/tokenizer-0-semscholar.pickle", 'rb') as handle:
            tokenizer = pickle.load(handle)
        sequences = tokenizer.texts_to_sequences(self.list_captions)
        data_text = pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")     
        self.tokenizer = tokenizer
        self.captions_processed = data_text
        lb = LabelBinarizer()
        self.labels_processed = lb.fit_transform(self.categories)        
class CaptionExperiment:
    def __init__(self, corpus, batchSize, dim, optimizer, epochs, weights=None, trainable=False):
        self.corpus = corpus
        self.batchSize = batchSize
        self.dim = dim
        self.optimizer = optimizer
        self.epochs = epochs
        self.weights = weights
        self.trainable = trainable
        self.X = None
        self.X = corpus.captions_processed
        self.Y = corpus.labels_processed
        self.num_class = 5
        self.n_images = len(corpus.list_captions)
        self.id_experiment = ""

    def generate_caption_model(self):
        self.id_experiment = "weights-"+ str(self.weights)
        modelCaptions = Sequential()
        modelCaptions.add(Embedding(len(self.corpus.tokenizer.word_index)+1, self.dim, embeddings_initializer="uniform", input_length=self.corpus.max_len,trainable=self.trainable))
        modelCaptions.add(Conv1D(512, 5, activation="relu",trainable=self.trainable))
        modelCaptions.add(MaxPooling1D(5,trainable=self.trainable))
        modelCaptions.add(Conv1D(512, 5, activation="relu",trainable=self.trainable))
        modelCaptions.add(MaxPooling1D(5,trainable=self.trainable))
        modelCaptions.add(Conv1D(512, 5, activation="relu",trainable=self.trainable))
        modelCaptions.add(MaxPooling1D(35,trainable=self.trainable))
        modelCaptions.add(Reshape((1, 1, 512),trainable=self.trainable))
        if self.weights == "FCC6":
            modelCaptions.load_weights('./saved/models/modelCaptions_weights_FCC6.h5')
        modelCaptions.add(Flatten())
        modelCaptions.add(Dense(128, activation='relu'))
        modelCaptions.add(Dense(self.num_class, activation='softmax'))

        return modelCaptions

    def caption_experiment(self):
        kfold = KFold(n_splits=10, shuffle=True)
        it = 1

        print ("Number of images: "+str(self.n_images))
        print ("Number of classes: "+str(self.num_class))

        for train, test in kfold.split([None] * self.n_images):
            print("FOLD: " +str(it))
            print("Training with "+ str(len(train))+ " samples and validating with "+str(len(test)))
            model = self.generate_caption_model()
            model.compile(loss="categorical_crossentropy",
                          optimizer=self.optimizer,
                          metrics=["categorical_accuracy"])
            model.fit(self.X[train],self.Y[train],
                      validation_data=[self.X[test],
                                       self.Y[test]],
                      batch_size = self.batchSize,
                      epochs=self.epochs)
            it=it+1
