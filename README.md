# Look, Read and Enrich

## Notebooks and materials to reproduce the experiments and results in: Look, Read and Enrich - Learning from Scientific Figures and their Captions.

Automatically interpreting scientific figures is particularly hard compared to natural images. However, there is a valuable source of information in scientific literature that until now has remained untapped: the correspondence between a figure and its caption. Here we 
introduce a Language-Vision Correspondence learning task that results from investigating what can be learnt by looking at a large number of figures and reading their captions and provide the necessary code and data to reproduce the experiments related to such task.

LVC trains visual and language networks without additional supervision other than pairs of unconstrained figures and captions. We also support transferring lexical and semantic knowledge from existing knowledge graphs, which has proved to significantly improve the resulting features. 

This repository also provides code and data to leverage the LVC visual and language features in transfer learning tasks involving scientific text and figures, namely classification and multi-modal machine comprehension. Upon execution, our experiments show improvement or results on par with supervised baselines and ad-hoc approaches.

_**Disclaimer:** This repo does not offer embeddings extracted from proprietary knowledge graphs in order to comply with their IP and licensing schemes. Soon, we will offer versions built on top of open source KGs like WordNet._

## How to run the notebooks:

**1. Execute the script download.py with the materials you want to download:**

```
python download.py [options]

Options:
  -h, --help        show this help message and exit
  --cross-scigraph  Cross-modal experiment with Scigraph corpus
  --cross-semantic  Cross-modal experiment with Semantic Scholar corpus
  --cat-captions    Categorization experiment with captions
  --cat-figures     Categorization experiment with figures
  --tqa             TQA experiment
```

**2. Use the different notebooks to execute the experiments.**

**CrossModal-Experiments**: In this notebook we execute a correspondance experiment between the scientific figures and their captions as they appear together in a scientific publication. The corpora used in this experiment can be Scigraph or Semantic Scholar.

**Categorization-Experiments**: In this notebook we categorize the figures and captions in five different categories. To do so, we use the weights generated in the CrossModal experiment for the captions and with the introduction of KG (Vecsigrafo) for the figures.

**TQA-Experiments**: In this notebook we reproduce the baseline for the TQA challenge, as we replace the VGG-19 with the visual network of the LVC (Language-Visual Correspondance) model to extract the features of the figures within the context and questions.

## Requirements:
~300 GB of free space disk to reproduce all the experiments:
- cross-scigraph -> 40 GB
- cross-semantic-scholar -> 215 GB
- cat-figures -> 20 GB
- cat-captions -> 3 GB
- tqa -> 6 GB
