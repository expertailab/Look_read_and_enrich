# Look, Read and Enrich

## Materials to reproduce the experiments and results in: Look, Read and Enrich - Learning from Scientific Figures and their Captions.

Automatically interpreting scientific figures is particularly hard compared to natural images. However, there is a valuable source of information in scientific literature that until now has remained untapped: the correspondence between a figure and its caption. In the associated paper we present a Figure-Caption Correspondence (FCC) learning task that results from investigating what can be learnt by looking at a large number of figures and reading their captions. In this repository we provide the necessary code and data to reproduce the experiments related to such task that are presented in the paper.

<img src="./architecture diagrams/2-branch_nn_FCC_arch_final.png" width="2000">

FCC trains visual and language networks without additional supervision other than pairs of unconstrained figures and captions. We also support transferring lexical and semantic knowledge from existing knowledge graphs, which has proved to significantly improve the resulting features. 

This repository also provides code and data to leverage the FCC visual and language features in transfer learning tasks involving scientific text and figures for multi-modal classification of scientific figures and text over a taxonomy and multi-modal machine comprehension for question answering. 

## Dependencies:
To use this code you will need:

* Python 3.6.3
* Keras 2.2.4
* Scikit-learn 0.20.1
* Numpy 1.16.1
* Pillow 5.4.1
* Tqdm 4.28.1
* ~25GB free space disk

## How to run the experiments:

**1. Execute the script download.py to download from [Zenodo](https://zenodo.org/record/3258126) the materials (corpora and weights):**

```
python download.py
```

**2. Use the different python scripts to execute the experiments.**

**Figure-Caption Correspondance Experiments**: We execute correspondence and bidirectional retrieval experiments between the scientific figures and their captions. The corpora used are Scigraph or Semantic Scholar. Also, for the bidirectional retrieval task we support [Coco](http://cocodataset.org/#download)(2014)  and [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/). To use Flickr30k/Coco, download the images from their repositories, resize them to 224x224 resolution, and leave the resulting images in folders "look_read_and_enrich/images/coco/" and "look_read_and_enrich/images/flickr30k/".

```
fcc.py [-h] -c CORPUS [-t]

optional arguments:
  -h, --help                    show this help message and exit
  -t, --trainable               Trainable Vision Model

required arguments:
  -c CORPUS, --corpus CORPUS    Selected Corpus: flickr30k, coco, scigraph or semscholar
```

**Multimodal Classification Experiments**: We categorize the figures and captions over the SciGraph categories. To do so, we use the visual features generated in the FCC experiment (FCC6: plain FCC trained on SemScholar, FCC7: includes semantic embeddings from Vecsigrafo).

```
multimodal_classification.py [-h] [-w WEIGHTS] [-t]

optional arguments:
  -h, --help                      show this help message and exit
  -w WEIGHTS, --weights WEIGHTS   Weights: FCC6 or FCC7
  -t, --trainable                 Trainable Model
```

**TQA Experiments**: We reproduce the baselines for the TQA challenge and extend it with the FCC visual features and pre-trained semantic embeddings from Vecsigrafo. Question types include multiple choice non-diagram questions and multiple choice diagram questions. Model types include text only, diagram only, Cross (using the visual FCC features) and CrossVecsi (adding Vecsigrafo to represent the TQA words).

```
tqa.py [-h] -q QUESTIONTYPE -m MODELTYPE

optional arguments:
  -h, --help                                      show this help message and exit

required arguments:
  -q QUESTIONTYPE, --questionType QUESTIONTYPE    Question Type: nonDiagramQuestions or diagramQuestions
  -m MODELTYPE, --modelType MODELTYPE             Model Type: Text, Diagram, Cross or CrossVecsi
```

## Qualitative analysis FCC task - Selected features and heatmaps
The individual figures selected for the analysis in the paper can be found in  the folder **qualitative_analysis** folder.

**Selected visual features**. We pick the features with the most significant activation over the test set and select the figures that
activate them most. The figure shows a selection of 6 visual features with the 4 figures that activate each feature more significantly. The top row labels the dominant pattern for each feature 
<img src="./qualitative_analysis/figures/visual_features.png" width="2000">

**Visual features (heatmaps)**. The figure below shows the activation heatmaps corresponding to the features above and the figures that activate them more significantly.
<img src="./qualitative_analysis/figures/visual_features_heatmaps.png" width="2000">

**Text features**. Similar to the visual case, we selected the features from the last block of the language subnetwork with the highest
activation. For visualization purposes, we picked the figures corresponding to the captions that most activate such
features.
<img src="./qualitative_analysis/captions/caption_features.png" width="2000">

**Text features (heatmaps)**
Sample caption activation heatmaps. Red is high activation, blue is low.

"The Aliev-Panfilov model with  α = 0.01, ɣ = 0.002, b = 0.15, c = 8, μ1 = 0.2, μ2 = 0.3. The phase portrait depicts trajectories for distinct initial values φ0 and r0 (filled circles) converging to a stable equilibrium point (top). Non-oscillatory normalized time plot of the non-dimensional action potential φ and the recovery variable r (bottom)."
<img src="./qualitative_analysis/captions/caption_heatmap_1.png" width="2000">
<img src="./qualitative_analysis/captions/caption_heatmap_2.png" width="2000">

