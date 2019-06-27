from keras.callbacks import History
import json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense, Concatenate, Add, Average, Input, InputLayer, Reshape, BatchNormalization, LSTM, Lambda, Permute, Maximum, Bidirectional
import tensorflow as tf
from keras import backend as K
from keras.applications.vgg19 import VGG19
from PIL import Image
from tqdm import tqdm
from collections import Counter

class Corpus:
    # Initializer / Instance Attributes
    def __init__(self, json_file, questionType, modelType, pareto = False):
        self.json_file = json_file
        self.questionType = questionType
        self.modelType = modelType
        self.pareto = pareto
        self.data = []
        self.images = []
        self.tokenizers = []
        self.max_lens = []
        self.vocab_sizes = []
        self.correct_answers = []
        self.word_indexes = []
        self.X = []
        self.y = []

    def get_data(self):
        list_paragraphs = []
        list_paragraphs_imgs = []
        list_questions = []
        list_questions_imgs = []
        list_a_answers = []
        list_b_answers = []
        list_c_answers = []
        list_d_answers = []
        with open(self.json_file, "r") as file:
            dataset = json.load(file)
        for doc in dataset:
            if ((self.questionType == "nonDiagramQuestions" and doc["question_img"] is not "") or (self.questionType == "diagramQuestions" and doc["question_img"] == "")):
                continue
            list_paragraphs.append(doc["paragraph"])
            if doc["paragraph_img"] == "":
                list_paragraphs_imgs.append(doc["paragraph_img"])
            else:
                list_paragraphs_imgs.append("./images/tqa/"+doc["paragraph_img"])
            list_questions.append(doc["question"])
            if doc["question_img"] == "":
                list_questions_imgs.append(doc["question_img"])
            else:
                list_questions_imgs.append("./images/tqa/"+doc["question_img"])
            list_a_answers.append(doc["answer_a"])
            list_b_answers.append(doc["answer_b"])
            list_c_answers.append(doc["answer_c"])
            list_d_answers.append(doc["answer_d"])
            correct_answer = doc["correct_answer"]
            correct_array = np.zeros(4)
            letter_list=["a","b","c","d"]
            for i in range(4):
                if letter_list[i]==correct_answer:
                    correct_array[i]=1
            self.correct_answers.append(correct_array)
        data_untokens = [list_paragraphs, list_questions, list_a_answers, list_b_answers, list_c_answers, list_d_answers]
        self.images = [list_paragraphs_imgs,list_questions_imgs]
        self.data = [data_untokens]

    def process_corpus(self):
        for dat in self.data:
            texts = []
            for d in dat:
                if self.pareto:
                    dict_count = Counter([len(x.split(" ")) for x in d])
                    threshold = 0.8*sum([len(x.split(" ")) for x in d])
                    sorted_by_key = sorted(dict_count.items(), key=lambda kv: kv[0])
                    cont = 0
                    for elem in sorted_by_key:
                        if cont >= threshold:
                            break
                        else:
                            max_len = elem[0]
                            cont = cont + elem[0]*elem[1]
                else:
                    max_len=max([len(x.split(" ")) for x in d])
                texts.extend(d)
                self.max_lens.append(max_len)
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(texts)
            word_index = tokenizer.word_index
            vocab_size = len(word_index)+1


            self.tokenizers.append(tokenizer)
            self.word_indexes.append(word_index)
            self.vocab_sizes.append(vocab_size)

    def get_sequences(self):
        for i in range(len(self.data[0])):
            seq_tokens = self.tokenizers[0].texts_to_sequences(self.data[0][i])
            self.X.append(pad_sequences(seq_tokens, maxlen=self.max_lens[i], padding="post", truncating="post"))
            if (self.modelType=="Diagram" or self.modelType=="Cross" or self.modelType=="CrossVecsi") and (i == 0 or i==1):
                feat_list =process_image(self.modelType,self.images[i])
                self.X.append(feat_list)
        self.y = np.array(self.correct_answers)
    def get_split_XY(self,train,test):
        X_train_out = [x[train] for x in self.X]
        X_test_out = [x[test] for x in self.X]
        return X_train_out,self.y[train],X_test_out,self.y[test]

def process_image(modelType,images):
    if modelType == "Diagram":
        initial_model = VGG19(weights="imagenet", include_top=False)
        last = initial_model.output
        x = MaxPooling2D(7)(last)
        modelImages = Model(initial_model.input, x)
    else:
        modelImages = Sequential()
        modelImages.add(InputLayer(input_shape=(224, 224, 3)))
        modelImages.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D(2))
        modelImages.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D(2))
        modelImages.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D(2))
        modelImages.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D((28, 28), 2))
        if modelType == "Cross":
            modelImages.load_weights('./saved/models/modelImages_weights_FCC6.h5')
        if modelType == "CrossVecsi":
            modelImages.load_weights('./saved/models/modelImages_weights_FCC7.h5')
            
    feat_list = []
    for img in tqdm(images,total=len(images)):
        if img != "":
            figure_file = open(img, 'rb')
            figure = Image.open(figure_file)
            figure_resized = figure.resize((224, 224), Image.ANTIALIAS)
            arr = np.array(figure_resized)
            figure.close()
            figure_file.close()
            feat_list.append(np.squeeze(modelImages.predict(np.expand_dims(arr, axis=0))))
        else:
            feat_list.append(np.zeros((512)))
    return np.array(feat_list)
class LossLearningRateScheduler(History):
    """
    A learning rate scheduler that relies on changes in loss function
    value to dictate whether learning rate is decayed or not.
    LossLearningRateScheduler has the following properties:
    base_lr: the starting learning rate
    lookback_epochs: the number of epochs in the past to compare with the loss function at the current epoch to determine if progress is being made.
    decay_threshold / decay_multiple: if loss function has not improved by a factor of decay_threshold * lookback_epochs, then decay_multiple will be applied to the learning rate.
    spike_epochs: list of the epoch numbers where you want to spike the learning rate.
    spike_multiple: the multiple applied to the current learning rate for a spike.
    """

    def __init__(self, base_lr, lookback_epochs, spike_epochs = None, spike_multiple = 10, decay_threshold = 0.002, decay_multiple = 0.5, loss_type = 'val_loss'):

        super(LossLearningRateScheduler, self).__init__()

        self.base_lr = base_lr
        self.lookback_epochs = lookback_epochs
        self.spike_epochs = spike_epochs
        self.spike_multiple = spike_multiple
        self.decay_threshold = decay_threshold
        self.decay_multiple = decay_multiple
        self.loss_type = loss_type


    def on_epoch_begin(self, epoch, logs=None):      
        if len(self.epoch) > self.lookback_epochs:
            current_lr = K.get_value(self.model.optimizer.lr)
            target_loss = self.history[self.loss_type] 
            loss_diff =  target_loss[-1-int(self.lookback_epochs)] - target_loss[-1]
            if loss_diff <= np.abs(target_loss[-1]) * (self.decay_threshold * self.lookback_epochs):
                K.set_value(self.model.optimizer.lr, current_lr * self.decay_multiple)
                current_lr = current_lr * self.decay_multiple
            if self.spike_epochs is not None and len(self.epoch) in self.spike_epochs:
                K.set_value(self.model.optimizer.lr, current_lr * self.spike_multiple)
        else:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        return K.get_value(self.model.optimizer.lr)

def similarity(x):
    return tf.matmul(x[0],x[1], transpose_a=True)

def output_similarity(input_shape):
    return (input_shape[0][0], input_shape[0][2], input_shape[1][2])

def reduce_max_layer(x):
    return tf.reduce_max(x, axis=2, keepdims=True)

def output_reduce_max_layer(input_shape):
    return (input_shape[0], input_shape[1], 1)

def answerer(x):
    return tf.multiply(x[1], x[0])

def output_answerer(input_shape):
    return (input_shape[1][0], input_shape[1][1], input_shape[1][2])

def reduce_sum_layer(x):
    return tf.reduce_sum(x, axis=2, keepdims=True)

def output_reduce_sum_layer(input_shape):
    return (input_shape[0], input_shape[1], 1)

class TQAModel:
    def __init__(self, dim, dout, rdout,modelType):
        self.dim = dim
        self.dout = dout
        self.rdout = rdout
        self.modelType = modelType
        self.model = None

    def generate_model(self, corpus):
        dim = self.dim
        modelType = self.modelType
        dout = self.dout
        rdout = self.dout
        vocab_sizes = corpus.vocab_sizes
        max_lens = corpus.max_lens
        if (modelType=="Text"):
            M_scratch_input = Input(shape = (max_lens[0],), name="M_scratch_input")
            U_scratch_input = Input(shape = (max_lens[1],), name="U_scratch_input")
            C1_scratch_input = Input(shape = (max_lens[2],), name="C1_scratch_input")
            C2_scratch_input = Input(shape = (max_lens[3],), name="C2_scratch_input")
            C3_scratch_input = Input(shape = (max_lens[4],), name="C3_scratch_input")
            C4_scratch_input = Input(shape = (max_lens[5],), name="C4_scratch_input")

            embedding_scratch_layer = Embedding(vocab_sizes[0], dim, embeddings_initializer="uniform", trainable=True)

            M_scratch_input_embedded = embedding_scratch_layer(M_scratch_input)
            U_scratch_input_embedded = embedding_scratch_layer(U_scratch_input)
            C1_scratch_input_embedded = embedding_scratch_layer(C1_scratch_input)
            C2_scratch_input_embedded = embedding_scratch_layer(C2_scratch_input)
            C3_scratch_input_embedded = embedding_scratch_layer(C3_scratch_input)
            C4_scratch_input_embedded = embedding_scratch_layer(C4_scratch_input)

            LSTM_layer = LSTM(units=dim, return_sequences=True, name="lstm_layer", dropout=dout, recurrent_dropout=rdout)

            M_input_encoded = LSTM_layer(M_scratch_input_embedded)
            M_input_encoded = Permute((2, 1), input_shape=(max_lens[0], dim,), name="M_permute")(M_input_encoded)
            modelM = Model(M_scratch_input, M_input_encoded)

            U_input_encoded = LSTM_layer(U_scratch_input_embedded)
            U_input_encoded = Permute((2, 1), input_shape=(max_lens[1], dim,), name="U_permute")(U_input_encoded)
            modelU = Model(U_scratch_input, U_input_encoded)

            C1_input_encoded = LSTM_layer(C1_scratch_input_embedded)
            C1_input_encoded = Permute((2, 1), input_shape=(max_lens[2], dim,), name="C1_permute")(C1_input_encoded)
            C1_input_encoded = Lambda(reduce_sum_layer, output_shape=output_reduce_sum_layer, name="C1_reduce_sum")(C1_input_encoded)
            modelC1 = Model(C1_scratch_input, C1_input_encoded)

            C2_input_encoded = LSTM_layer(C2_scratch_input_embedded)
            C2_input_encoded = Permute((2, 1), input_shape=(max_lens[3], dim,), name="C2_permute")(C2_input_encoded)
            C2_input_encoded = Lambda(reduce_sum_layer, output_shape=output_reduce_sum_layer, name="C2_reduce_sum")(C2_input_encoded)
            modelC2 = Model(C2_scratch_input, C2_input_encoded)

            C3_input_encoded = LSTM_layer(C3_scratch_input_embedded)
            C3_input_encoded = Permute((2, 1), input_shape=(max_lens[4], dim,), name="C3_permute")(C3_input_encoded)
            C3_input_encoded = Lambda(reduce_sum_layer, output_shape=output_reduce_sum_layer, name="C3_reduce_sum")(C3_input_encoded)
            modelC3 = Model(C3_scratch_input, C3_input_encoded)

            C4_input_encoded = LSTM_layer(C4_scratch_input_embedded)
            C4_input_encoded = Permute((2, 1), input_shape=(max_lens[5], dim,), name="C4_permute")(C4_input_encoded)
            C4_input_encoded = Lambda(reduce_sum_layer, output_shape=output_reduce_sum_layer, name="C4_reduce_sum")(C4_input_encoded)
            modelC4 = Model(C4_scratch_input, C4_input_encoded)

            modelIna = Lambda(similarity, output_shape=output_similarity,name="similarityMU")([modelM.output, modelU.output])
            modelIna = Lambda(reduce_max_layer, output_shape=output_reduce_max_layer, name="a_reduce_max")(modelIna)
            modelIna = Permute((2, 1), input_shape=(max_lens[0], 1,), name="a_permute")(modelIna)
            modelIna = Dense(max_lens[0],activation="softmax",name="softmax_a")(modelIna)
            modela = Model([M_scratch_input, U_scratch_input], modelIna)

            modelInm = Lambda(answerer, output_shape=output_answerer,name="answerer") ([modela.output, modelM.output])
            modelInm = Lambda(reduce_sum_layer, output_shape=output_reduce_sum_layer, name="m_reduce_sum") (modelInm)
            modelm = Model([M_scratch_input, U_scratch_input], modelInm)

            modelInC1 = Lambda(similarity, output_shape=output_similarity,name="similaritymC1")([modelm.output,modelC1.output])
            modelIn1 = Model([M_scratch_input, U_scratch_input, C1_scratch_input], modelInC1)
            modelInC2 = Lambda(similarity, output_shape=output_similarity,name="similaritymC2")([modelm.output,modelC2.output])
            modelIn2 = Model([M_scratch_input, U_scratch_input, C2_scratch_input], modelInC2)
            modelInC3 = Lambda(similarity, output_shape=output_similarity,name="similaritymC3")([modelm.output,modelC3.output])
            modelIn3 = Model([M_scratch_input, U_scratch_input, C3_scratch_input], modelInC3)
            modelInC4 = Lambda(similarity, output_shape=output_similarity,name="similaritymC4")([modelm.output,modelC4.output])
            modelIn4 = Model([M_scratch_input, U_scratch_input,C4_scratch_input], modelInC4)

            modelIn = Concatenate(name = "concatenate")([modelIn1.output,modelIn2.output,modelIn3.output,modelIn4.output])
            modelIn = Flatten()(modelIn)
            modelIn = Dense(4, activation="softmax",name="softmax_y") (modelIn)
            model = Model([M_scratch_input, U_scratch_input, C1_scratch_input, C2_scratch_input, C3_scratch_input, C4_scratch_input], modelIn)
        if modelType == "Diagram" or modelType == "Cross" or modelType == "CrossVecsi":
            M_scratch_input = Input(shape=(max_lens[0],), name="M_scratch_input")
            U_scratch_input = Input(shape=(max_lens[1],), name="U_scratch_input")
            C1_scratch_input = Input(shape=(max_lens[2],), name="C1_scratch_input")
            C2_scratch_input = Input(shape=(max_lens[3],), name="C2_scratch_input")
            C3_scratch_input = Input(shape=(max_lens[4],), name="C3_scratch_input")
            C4_scratch_input = Input(shape=(max_lens[5],), name="C4_scratch_input")

            modelMF = Sequential()
            modelMF.add(InputLayer(input_shape=(512,), name="input_MF"))
            modelMF.add(Dense(256, activation="tanh", name="perceptron_MF_1"))
            modelMF.add(Dense(dim, activation="tanh", name="perceptron_MF_2"))
            modelMF.add(Reshape((dim, 1,), name="reshape_MF"))
            modelUF = Sequential()
            modelUF.add(InputLayer(input_shape=(512,), name="input_UF"))
            modelUF.add(Dense(256, activation="tanh", name="perceptron_UF_1"))
            modelUF.add(Dense(dim, activation="tanh", name="perceptron_UF_2"))
            modelUF.add(Reshape((dim, 1,), name="reshape_UF"))

            embedding_scratch_layer = Embedding(vocab_sizes[0], dim, embeddings_initializer="uniform",
                                                    trainable=True)

            M_input_embedded = embedding_scratch_layer(M_scratch_input)
            U_input_embedded = embedding_scratch_layer(U_scratch_input)
            C1_input_embedded = embedding_scratch_layer(C1_scratch_input)
            C2_input_embedded = embedding_scratch_layer(C2_scratch_input)
            C3_input_embedded = embedding_scratch_layer(C3_scratch_input)
            C4_input_embedded = embedding_scratch_layer(C4_scratch_input)

            LSTM_layer = LSTM(units=dim, return_sequences=True, name="lstm_layer", dropout=dout, recurrent_dropout=rdout)

            M_input_encoded = LSTM_layer(M_input_embedded)
            M_input_encoded = Permute((2, 1), input_shape=(max_lens[0], dim,), name="M_permute")(M_input_encoded)
            modelM = Model(M_scratch_input, M_input_encoded)
            modelInMMF = Concatenate(name="concatenateMMF")([modelM.output, modelMF.output])
            modelMMF = Model([M_scratch_input, modelMF.input], modelInMMF)

            U_input_encoded = LSTM_layer(U_input_embedded)
            U_input_encoded = Permute((2, 1), input_shape=(max_lens[1], dim,), name="U_permute")(U_input_encoded)
            modelU = Model(U_scratch_input, U_input_encoded)
            modelInUUF = Concatenate(name="concatenateUUF")([modelU.output, modelUF.output])
            modelUUF = Model([U_scratch_input, modelUF.input], modelInUUF)

            C1_input_encoded = LSTM_layer(C1_input_embedded)
            C1_input_encoded = Permute((2, 1), input_shape=(max_lens[2], dim,), name="C1_permute")(C1_input_encoded)
            C1_input_encoded = Lambda(reduce_sum_layer, output_shape=output_reduce_sum_layer, name="C1_reduce_sum")(C1_input_encoded)
            modelC1 = Model(C1_scratch_input, C1_input_encoded)

            C2_input_encoded = LSTM_layer(C2_input_embedded)
            C2_input_encoded = Permute((2, 1), input_shape=(max_lens[3], dim,), name="C2_permute")(C2_input_encoded)
            C2_input_encoded = Lambda(reduce_sum_layer, output_shape=output_reduce_sum_layer, name="C2_reduce_sum")(C2_input_encoded)
            modelC2 = Model(C2_scratch_input, C2_input_encoded)

            C3_input_encoded = LSTM_layer(C3_input_embedded)
            C3_input_encoded = Permute((2, 1), input_shape=(max_lens[4], dim,), name="C3_permute")(C3_input_encoded)
            C3_input_encoded = Lambda(reduce_sum_layer, output_shape=output_reduce_sum_layer, name="C3_reduce_sum")(C3_input_encoded)
            modelC3 = Model(C3_scratch_input, C3_input_encoded)

            C4_input_encoded = LSTM_layer(C4_input_embedded)
            C4_input_encoded = Permute((2, 1), input_shape=(max_lens[5], dim,), name="C4_permute")(C4_input_encoded)
            C4_input_encoded = Lambda(reduce_sum_layer, output_shape=output_reduce_sum_layer, name="C4_reduce_sum")(C4_input_encoded)
            modelC4 = Model(C4_scratch_input, C4_input_encoded)

            modelIna = Lambda(similarity, output_shape=output_similarity, name="similarityMU")([modelMMF.output, modelUUF.output])
            modelIna = Lambda(reduce_max_layer, output_shape=output_reduce_max_layer, name="a_reduce_max")(modelIna)
            modelIna = Permute((2, 1), input_shape=(max_lens[0] + 1, 1,), name="a_permute")(modelIna)
            modelIna = Dense(max_lens[0] + 1, activation="softmax", name="softmax_a")(modelIna)
            modela = Model([M_scratch_input, modelMF.input, U_scratch_input, modelUF.input], modelIna)

            modelInm = Lambda(answerer, output_shape=output_answerer, name="answerer")([modela.output, modelMMF.output])
            modelInm = Lambda(reduce_sum_layer, output_shape=output_reduce_sum_layer, name="m_reduce_sum")(modelInm)
            modelm = Model([M_scratch_input, modelMF.input, U_scratch_input, modelUF.input], modelInm)

            modelInC1 = Lambda(similarity, output_shape=output_similarity, name="similaritymC1")([modelm.output, modelC1.output])
            modelIn1 = Model([M_scratch_input, modelMF.input,U_scratch_input, modelUF.input,C1_scratch_input], modelInC1)
            modelInC2 = Lambda(similarity, output_shape=output_similarity, name="similaritymC2")([modelm.output, modelC2.output])
            modelIn2 = Model([M_scratch_input, modelMF.input,U_scratch_input, modelUF.input,C2_scratch_input], modelInC2)
            modelInC3 = Lambda(similarity, output_shape=output_similarity, name="similaritymC3")([modelm.output, modelC3.output])
            modelIn3 = Model([M_scratch_input, modelMF.input,U_scratch_input, modelUF.input,C3_scratch_input], modelInC3)
            modelInC4 = Lambda(similarity, output_shape=output_similarity, name="similaritymC4")([modelm.output, modelC4.output])
            modelIn4 = Model([M_scratch_input, modelMF.input,U_scratch_input, modelUF.input,C4_scratch_input], modelInC4)

            modelIn = Concatenate(name="concatenate")([modelIn1.output, modelIn2.output, modelIn3.output, modelIn4.output])
            modelIn = Flatten()(modelIn)
            modelIn = Dense(4, activation="softmax", name="softmax_y")(modelIn)
            model = Model([M_scratch_input, modelMF.input,U_scratch_input, modelUF.input,C1_scratch_input, C2_scratch_input, C3_scratch_input,C4_scratch_input], modelIn)
        return model
