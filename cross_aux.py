import random
import sys
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold, train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense, Concatenate, Add, InputLayer, Reshape,BatchNormalization, Dropout, Multiply, LSTM
from keras.models import Model, Sequential

class Corpus:
    def __init__(self, corpus_selected, visionTrainable):
        self.corpus_selected = corpus_selected
        self.visionTrainable = visionTrainable
        self.tokenizer = None
        self.list_captions = []
        self.captions_processed = []
        self.list_images = []
        self.images_processed = None
        self.n_captions = 0
        self.max_len = 1000
        vgg16_model = VGG16(weights="imagenet", include_top=True)
        self.modelPre = Sequential()
        for layer in vgg16_model.layers[:-1]:
            self.modelPre.add(layer)

    def generate_corpus(self):
        with open("./jsons/"+self.corpus_selected+".json", "r", encoding="utf-8", errors="surrogatepass") as file:
            dataset = json.load(file)
        for doc in dataset:
            self.list_captions.extend(doc["captions"])
            self.n_captions = len(doc["captions"])
            self.list_images.append("./images/"+self.corpus_selected+"/"+
                                    doc["img_file"])
    def process_corpus(self): 
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(list_captions)
        sequences = tokenizer.texts_to_sequences(list_captions)
        data_text = pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")     
        self.tokenizer = tokenizer
        self.captions_processed[cont] = self.group_txt(data_text)
        self.images_processed = [self.process_image(x) for x in tqdm(self.list_images, total=len(self.list_images))]
    def group_txt (self, data_text):
        cont = 0
        res = []
        for txt in data_text:
            cont = cont+1
            if cont == 1:
                aux_list = []
            aux_list.append(txt)
            if cont == self.n_captions:
                res.append(aux_list)
                aux_list = []
                cont = 0
        return res
    def process_image (self, image):
        if self.visionTrainable and self.corpus_selected == "semscholar":
            return image
        f_img = open(image, 'rb')
        im = Image.open(f_img)
        arr = np.array(im.convert(mode='RGB'))
        im.close()
        f_img.close()
        if self.visionTrainable and self.corpus_selected != "semscholar":
            return arr
        if self.visionTrainable == False:
            return np.squeeze(self.modelPre.predict(np.expand_dims(arr,axis=0)))
        
class CrossExperiment:
    def __init__(self, corpus, batchSize, dim, optimizer, epochs):
        self.corpus = corpus
        self.batchSize = batchSize
        self.dim = dim
        self.optimizer = optimizer
        self.epochs = epochs
        self.captions_types = None
        self.num_class = 2
        self.n_images = len(corpus.list_images)

    def generate_model(self):
        modelImages = Sequential()
        modelCaptions = Sequential()
        modelCaptions.add(Embedding(len(self.corpus.tokenizers[0].word_index)+1, self.dim, embeddings_initializer="uniform", input_length=self.corpus.max_len,trainable=True))
        modelCaptions.add(Conv1D(512, 5, activation="relu"))
        modelCaptions.add(MaxPooling1D(5))
        modelCaptions.add(Conv1D(512, 5, activation="relu"))
        modelCaptions.add(MaxPooling1D(5))
        modelCaptions.add(Conv1D(512, 5, activation="relu"))
        modelCaptions.add(MaxPooling1D(35))
        if self.corpus.visionTrainable == False:
            modelCaptions.add(Flatten())
        else:
            modelCaptions.add(Reshape((1, 1, 512)))

        if self.corpus.visionTrainable:
            modelImages.add(InputLayer(input_shape=(224,224,3)))
            modelImages.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
            modelImages.add(BatchNormalization())
            modelImages.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
            modelImages.add(BatchNormalization())
            modelImages.add(MaxPooling2D(2))
            modelImages.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
            modelImages.add(BatchNormalization())
            modelImages.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
            modelImages.add(BatchNormalization())
            modelImages.add(MaxPooling2D(2))
            modelImages.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
            modelImages.add(BatchNormalization())
            modelImages.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
            modelImages.add(BatchNormalization())
            modelImages.add(MaxPooling2D(2))
            modelImages.add(Conv2D(512, (3,3), padding = "same", activation="relu"))
            modelImages.add(BatchNormalization())
            modelImages.add(Conv2D(512, (3,3), padding = "same", activation="relu")) 
            modelImages.add(BatchNormalization())
            modelImages.add(MaxPooling2D((28,28),2))

        if self.corpus.visionTrainable == False:
            modelImages.add(InputLayer(input_shape=(4096,)))
            modelImages.add(Dense(2048, activation='relu'))
            modelImages.add(Dense(512, activation='relu'))

        mergedOut = Multiply()([modelCaptions.output,modelImages.output])  

        if self.corpus.visionTrainable:
            mergedOut = Flatten()(mergedOut)
            
        mergedOut = Dense(128, activation='relu')(mergedOut)
        mergedOut = Dense(2, activation='softmax')(mergedOut)
        model = Model([modelCaptions.input,modelImages.input], mergedOut)
        return model

    def correspondance_experiment(self):
        kfold = KFold(n_splits=10, shuffle=True)
        it = 1

        print ("Number of images: "+str(self.n_images))
        print ("Number of classes: "+str(self.num_class))

        for train, test in kfold.split([None] * self.n_images):
            print("FOLD: " +str(it))
            print("Training with "+ str(len(train))+ " samples and validating with "+str(len(test)))
            model = self.generate_model()
            model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=["categorical_accuracy"])
            model.fit_generator(self.generator(train), epochs=self.epochs,
                        steps_per_epoch = len(train)//self.batchSize,
                        validation_data=(self.generator(test)),
                        validation_steps= len(test)//self.batchSize)
            it=it+1

    def ic_retrieval(self):
        indexes = list(range(self.n_images))
        train, test = train_test_split(indexes, test_size=1000/self.n_images)
        print("Entrenando con "+ str(len(train))+ " muestras y evaluando con "+str(len(test)))
        model = self.generate_model()
        model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=["categorical_accuracy"])
        model.fit_generator(self.generator(train), epochs=self.epochs,
                        steps_per_epoch = len(train)//self.batchSize,
                        validation_data=(self.generator(test)),
                        validation_steps= len(test)//self.batchSize)
        print("\n")
        self.caption_retrieval(model,test)
        print("\n")
        self.image_retrieval(model,test)
        print("\n")

    def generator(self, indexes):
        while True:
            np.random.shuffle(indexes)
            for i in range(0, len(indexes), self.batchSize):
                batch_indexes = indexes[i:i+self.batchSize]
                batch_indexes.sort()

                bx1,bx2,by = self.get_batches(batch_indexes)

                yield ([bx1,bx2], by)

    def get_batches(self, batch_indexes):
        tuples_to_shuffle = []
        for batch_ind in batch_indexes:
            for ind_caption in range(self.corpus.n_captions):
                tuple1 = []
                tuple1.append(self.corpus.captions_processed[batch_ind][ind_caption])
                if self.corpus.corpus_selected=="semscholar" and self.corpus.visionTrainable:
                    f_img = open(self.corpus.images_processed[batch_ind], 'rb')
                    im = Image.open(f_img)
                    arr = np.array(im)
                    im.close()
                    f_img.close()
                    tuple1.append(arr)
                else:
                    tuple1.append(self.corpus.images_processed[batch_ind])
                tuple1.append(np.array([0,1]))
                tuples_to_shuffle.append(tuple1)
                rand_ind = random.choice([x for x in range(self.n_images) if x != batch_ind])
                rand_subind = random.randint(0, self.corpus.n_captions-1)
                tuple2 = []
                tuple2.append(self.corpus.captions_processed[rand_ind][rand_subind])
                if self.corpus.corpus_selected=="semscholar" and self.corpus.visionTrainable:
                    f_img = open(self.corpus.images_processed[batch_ind], 'rb')
                    im = Image.open(f_img)
                    arr = np.array(im)
                    im.close()
                    f_img.close()
                    tuple2.append(arr)
                else:
                    tuple2.append(self.corpus.images_processed[batch_ind])
                tuple2.append(np.array([1,0]))
                tuples_to_shuffle.append(tuple2)
        random.shuffle(tuples_to_shuffle)
        x1 = []
        x2 = []
        y = []
        for i in tuples_to_shuffle:
            x1.append(i[0])
            x2.append(i[1])
            y.append(i[2])
        return np.array(x1),np.array(x2),np.array(y)

    def image_retrieval(self, model, test):
        r_at_1 = 0
        r_at_5 = 0
        r_at_10 = 0
        cont = 0
        for i in tqdm(test,total=len(test)):
            for ind_capt in range(self.corpus.n_captions):
                bx1 = np.expand_dims(self.corpus.captions_processed[i][ind_capt],axis=0)
                if self.corpus.corpus_selected=="semscholar" and self.corpus.visionTrainable:
                    f_img = open(self.corpus.images_processed[i], 'rb')
                    im = Image.open(f_img)
                    arr = np.array(im)
                    im.close()
                    f_img.close()
                    bx2 = np.expand_dims(arr,axis=0))
                else:
                    bx2 = np.expand_dims(self.corpus.images_processed[i],axis=0))
                count_cand = 0
                good_pred = model.predict([bx1,bx2])
                for j in test:
                    if i == j:
                        continue
                    if self.corpus.corpus_selected=="semscholar" and self.corpus.visionTrainable:
                        f_img = open(self.corpus.images_processed[j], 'rb')
                        im = Image.open(f_img)
                        arr = np.array(im)
                        im.close()
                        f_img.close()
                        bx2 = np.expand_dims(arr,axis=0)
                    else:
                        bx2 = np.expand_dims(self.corpus.images_processed[j],axis=0)
                    cand_pred = model.predict([bx1,bx2])
                    if cand_pred[:,1] > good_pred[:,1]:
                        count_cand = count_cand + 1
                    if count_cand >= 10:
                        break
                if count_cand < 10:
                    r_at_10 = r_at_10 + 1
                if count_cand < 5:
                    r_at_5 = r_at_5 + + 1
                if count_cand < 1:
                    r_at_1 = r_at_1 + 1
        print ("IMAGE RETRIEVAL (r@1: {} r@5: {} r@10: {}".format(r_at_1/(len(test)*self.corpus.n_captions), r_at_5/(len(test)*self.corpus.n_captions), r_at_10/(len(test)*self.corpus.n_captions)))
                                       
    def caption_retrieval(self, model, test):
        r_at_1 = 0
        r_at_5 = 0
        r_at_10 = 0
        for i in tqdm(test,total=len(test)):
            good_preds = []
            for ind_capt in range(self.corpus.n_captions):
                bx1 = np.expand_dims(self.corpus.captions_processed[i][ind_capt],axis=0)
                if self.corpus.corpus_selected=="semscholar" and self.corpus.visionTrainable:
                    f_img = open(self.corpus.images_processed[i], 'rb')
                    im = Image.open(f_img)
                    arr = np.array(im)
                    im.close()
                    f_img.close()
                    bx2 = np.expand_dims(arr,axis=0)
                else:
                    bx2 = np.expand_dims(self.corpus.images_processed[i],axis=0))
                good_preds.append(model.predict([bx1,bx2]))
            good_pred = good_preds[np.argmax([x[:,1] for x in good_preds])]
            count_cand = 0
            for j in test:
                if i == j:
                    continue
                for ind_capt in range(self.corpus.n_captions):
                    bx2 = np.expand_dims(self.corpus.captions_processed[j][ind_capt],axis=0))
                    if self.corpus.corpus_selected=="semscholar" and self.corpus.visionTrainable:
                        f_img = open(self.corpus.images_processed[i], 'rb')
                        im = Image.open(f_img)
                        arr = np.array(im)
                        im.close()
                        f_img.close()
                        bx2 = np.expand_dims(arr,axis=0)
                    else:
                        bx2 = np.expand_dims(self.corpus.images_processed[i],axis=0)
                    cand_pred = model.predict([bx1,bx2])
                    if cand_pred[:,1] > good_pred[:,1]:
                        count_cand = count_cand + 1
                if count_cand >= 10:
                    break
            if count_cand < 10:
                r_at_10 = r_at_10 + 1
            if count_cand < 5:
                r_at_5 = r_at_5 + + 1
            if count_cand < 1:
                r_at_1 = r_at_1 + 1
        print ("CAPTION RETRIEVAL (r@1: {} r@5: {} r@10: {}".format(r_at_1/len(test), r_at_5/len(test), r_at_10/len(test)))
