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

class FigureCorpus:
    def __init__(self):
        self.list_images = []
        self.categories = []
        self.labels_processed = None
        self.max_len = 1000

    def generate_corpus(self):
        with open("./jsons/scigraph.json", "r", encoding="utf-8", errors="surrogatepass") as file:
            dataset = json.load(file)
        for doc in dataset:
            self.list_images.append("./images/scigraph/"+doc["img_file"])
            self.categories.append(doc["category"])
    def process_corpus(self): 
        lb = LabelBinarizer()
        self.labels_processed = lb.fit_transform(self.categories)

class FigureExperiment:
    def __init__(self, corpus, batchSize, optimizer, epochs, weights=None, trainable=False):
        self.corpus = corpus
        self.batchSize = batchSize
        self.optimizer = optimizer
        self.epochs = epochs
        self.weights = weights
        self.trainable = trainable
        self.num_class = 5
        self.n_images = len(corpus.list_images)
        self.id_experiment = ""

    def generate_image_model(self):
        modelImages = Sequential()
        modelImages.add(InputLayer(input_shape=(224,224,3)))
        modelImages.add(Conv2D(64, (3,3), padding = "same", activation="relu",trainable=self.trainable))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(64, (3,3), padding = "same", activation="relu",trainable=self.trainable))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D(2,trainable=self.trainable))
        modelImages.add(Conv2D(128, (3,3), padding = "same", activation="relu",trainable=self.trainable))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(128, (3,3), padding = "same", activation="relu",trainable=self.trainable))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D(2,trainable=self.trainable))
        modelImages.add(Conv2D(256, (3,3), padding = "same", activation="relu",trainable=self.trainable))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(256, (3,3), padding = "same", activation="relu",trainable=self.trainable))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D(2,trainable=self.trainable))
        modelImages.add(Conv2D(512, (3,3), padding = "same", activation="relu",trainable=self.trainable))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(512, (3,3), padding = "same", activation="relu",trainable=self.trainable)) 
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D((28,28),2,trainable=self.trainable))
        if self.weights is not None:
            modelImages.load_weights('./saved/models/modelImages_weights_'+self.weights+'.h5')
        modelImages.add(Flatten())
        modelImages.add(Dense(128, activation='relu'))
        modelImages.add(Dense(self.num_class, activation='softmax'))

        return modelImages

    def figures_experiment(self):
        kfold = KFold(n_splits=10, shuffle=True)
        it = 1

        print ("Number of images: "+str(self.n_images))
        print ("Number of classes: "+str(self.num_class))

        for train, test in kfold.split([None] * self.n_images):
            print("FOLD: " +str(it))
            print("Training with "+ str(len(train))+ " samples and validating with "+str(len(test)))
            model = self.generate_image_model()
            model.compile(loss="categorical_crossentropy",
                          optimizer=self.optimizer,
                          metrics=["categorical_accuracy"])
            model.fit_generator(self.generator(train), epochs=self.epochs,
                        steps_per_epoch = len(train)//self.batchSize,
                        validation_data=(self.generator(test)),
                        validation_steps= len(test)//self.batchSize)
            it=it+1

    def generator(self, indexes):
        while True:
            np.random.shuffle(indexes)
            for i in range(0, len(indexes), self.batchSize):
                batch_indexes = indexes[i:i+self.batchSize]
                batch_indexes.sort()

                bx,by = self.get_batches(batch_indexes)

                yield np.array(bx), np.array(by)

    def get_batches(self, batch_indexes):
        bx = []
        by = []
        for batch_ind in batch_indexes:
            f_img = open(self.corpus.list_images[batch_ind], 'rb')
            im = Image.open(f_img)
            arr = np.array(im)
            im.close()
            f_img.close()
            bx.append(arr)
            by.append(self.corpus.labels_processed[batch_ind])
        return bx,by
        

