import sys
import argparse
from cat_cap_aux import CaptionCorpus, CaptionExperiment
from cat_fig_aux import FigureCorpus, FigureExperiment
from keras import optimizers

def main(argv):
    weights = None
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-w', '--weights', help='Weights: FCC6 or FCC7')
    parser.add_argument('-t', '--trainable', help='Trainable Model', action='store_true')
    args = parser.parse_args()

    
    if args.weights is not None:
        weights = args.weights

    trainable = args.trainable

    print("Executing the experiment with: " +
          "\n Weights: " + str(weights) +
          "\n Trainable: " + str(trainable))


    batchSize = 128
    epochs = 5
    dim = 300
    adam = optimizers.Adam()

    captionCorpus = CaptionCorpus()
    captionCorpus.generate_corpus()
    captionCorpus.process_corpus()
    
    captionExperiment = CaptionExperiment (captionCorpus, batchSize, dim, adam, epochs, weights, trainable)
    captionExperiment.caption_experiment()

    batchSize = 32
    epochs = 6
    adam = optimizers.Adam(lr=1e-4,decay=1e-5)

    figureCorpus = FigureCorpus()
    figureCorpus.generate_corpus()
    figureCorpus.process_corpus()
    
    figureExperiment = FigureExperiment (figureCorpus, batchSize, adam, epochs, weights, trainable)
    figureExperiment.figures_experiment()

if __name__ == "__main__":
   main(sys.argv[1:])
