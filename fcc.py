import sys
import argparse
from scripts.fcc_aux import Corpus, CrossExperiment
from keras import optimizers

def main(argv):
    corpus_selected = ""
    kg_emb = None
    visionTrainable = False

    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--corpus', help='Selected Corpus: flickr30k, coco, scigraph or semscholar', required=True)
    parser.add_argument('-t', '--trainable', help='Trainable Vision Model', action='store_true')
    args = parser.parse_args()

    batchSize = 16
    epochs = 4
    dim = 300
    adam = optimizers.Adam(lr=1e-4, decay=1e-5)

    corpus_selected = args.corpus
    visionTrainable = args.trainable

    print("Executing the experiment with: " +
          "\n Corpus: " + corpus_selected +
          "\n visionTrainable: " + str(visionTrainable))
    corpus = Corpus(corpus_selected, visionTrainable)
    corpus.generate_corpus()
    corpus.process_corpus()
    if corpus_selected == "flickr30k" or corpus_selected == "coco":
        batchSize = 3
    crossExperiment = CrossExperiment (corpus, batchSize, dim, adam, epochs)
    crossExperiment.correspondance_experiment()
    crossExperiment.ic_retrieval()

if __name__ == "__main__":
   main(sys.argv[1:])
