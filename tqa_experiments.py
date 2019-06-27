from tqa_aux import Corpus,TQAModel,LossLearningRateScheduler
from sklearn.model_selection import KFold
from keras import optimizers
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import sys
import argparse


def main(argv):
    weights = None
    
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-q', '--questionType', help='Question Type: nonDiagramQuestions or diagramQuestions', required=True)
    required.add_argument('-m', '--modelType', help='Model Type: Text, Diagram, Cross or CrossVecsi', required=True)
    args = parser.parse_args()

    questionType = args.questionType
    modelType = args.modelType

    print("Executing the experiment with: " +
          "\n QuestionType: " + questionType +
          "\n ModelType: " + modelType)

    if questionType == "nonDiagramQuestions":
        dout=0.5
        rdout=0.5
        pareto=False
    else:
        dout=0.0
        rdout=0.0
        pareto=True


    json_file = "./jsons/tqa.json"
    dim = 100
    batch_size = 128
    epochs = 5
    first_lr = 1e-2
    scheduling = False
    steps_sched = 1
    patience = 2

    corpus = Corpus(json_file,questionType,modelType, pareto)
    corpus.get_data()
    corpus.process_corpus()
    corpus.get_sequences()

    kfold = KFold(n_splits=10, shuffle=True)
    precisions = []
    recalls = []
    f1s = []
    it = 1

    for train, test in kfold.split(list(range(len(corpus.y)))):
        print("Iteration: "+ str(it))
        X_train,y_train,X_test,y_test = corpus.get_split_XY(train,test)
        tqamodel = TQAModel(dim, dout, rdout,modelType)
        model = tqamodel.generate_model(corpus)

        adam = optimizers.Adam(lr=first_lr)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['categorical_accuracy'])

        if (scheduling == True):
            lr_sched = LossLearningRateScheduler(first_lr,steps_sched)
            early_stop = EarlyStopping(patience=patience)
            history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose=1,
                            callbacks=[lr_sched,early_stop])
        else:
            history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose=1)
    
        pred = model.predict(X_test, batch_size=batch_size)
        max_value = np.argmax(pred,axis=1)
        predNew = np.zeros(np.shape(pred))
        for i in range(len(predNew)):
          predNew[i,max_value[i]]=1
        print(classification_report(y_test, predNew, digits=4, target_names=["a","b","c","d"]))
        precisions.append(precision_score(y_test, predNew, average="weighted"))
        recalls.append(recall_score(y_test, predNew, average="weighted"))
        f1s.append(f1_score(y_test, predNew, average="weighted"))
        it = it+1
    
    print("Precision: %.4f (+/- %.2f)" % (np.mean(precisions), np.std(precisions)))
    print("Recall: %.4f (+/- %.2f)" % (np.mean(recalls), np.std(recalls)))
    print("F1 Score: %.4f (+/- %.2f)" % (np.mean(f1s), np.std(f1s)))
    

if __name__ == "__main__":
   main(sys.argv[1:])

