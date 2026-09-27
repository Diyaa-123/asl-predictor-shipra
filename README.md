# ASL-Alphabet Image Classification

This small project contains approaches to classify letter/alphabet images that contain gestures of the American Sign Language (ASL). Deep Learning models are implemented using Keras, including (1) CNNs defined from scratch, (2) transfer learning with models pre-trained on ImageNet and (3) autoencoders in combination with random forests.

**TLDR**: The CNN defined and trained from scratch reaches an accuracy of `0.998` on the test set! Probably, and in part, because the dataset is easy.

Notes:

- The project repository can be found here: [asl_alphabet_image_classification](https://github.com/mxagar/asl_alphabet_image_classification).
- Some of the starter code and examples were taken from a [Datacamp](https://www.datacamp.com) guided project: [ASL Recognition with Deep Learning](https://app.datacamp.com/learn/projects/509).
- :warning: This is an experimental project where some Keras functionalities related to CNNs are showcased; however, these are not systematically applied to find the optimum solution. The motivation and performance of the models is discussed in a dedicated [section](#discussion-on-the-used-models), and at the end a non-exhaustive list of [possible improvements](#next-steps-improvements) is provided.

## Table of Contents

- [ASL-Alphabet Image Classification](#asl-alphabet-image-classification)
  - [Table of Contents](#table-of-contents)
  - [The Dataset](#the-dataset)
  - [How to Use This](#how-to-use-this)
    - [Dependencies](#dependencies)
  - [Discussion on the Used Models](#discussion-on-the-used-models)
    - [CNN from Scratch](#cnn-from-scratch)
    - [Fine-Tuning and Transfer Learning of ResNet50 and VGG16](#fine-tuning-and-transfer-learning-of-resnet50-and-vgg16)
    - [Autoencoder Compression + Random Forest](#autoencoder-compression--random-forest)
  - [Preliminary Conclusions](#preliminary-conclusions)
  - [Next Steps, Improvements](#next-steps-improvements)
  - [Authorship](#authorship)

## The Dataset

The original dataset can be downloaded from Kaggle: [asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). It consists of `50 x 50 px` color images in which gestures of the `A-Z` symbols are displayed with hands on different background and lighting conditions. 

![ASL Examples](./assets/asl_examples_labelled.jpg)

For this project, special symbols were ignored (i.e., `nothing`, `space`, `del`) and only the original `train` split was taken. Then, that `train` split was further segregated into the `train` and `test` subsets, with the following sizes:

- `train`: 60,000 observations
- `test`: 15,000 observations (20%)
- labels: 25 (`A-Z`)

The dataset is well balanced: each character has around 2,400 observations in the `train` split and around 600 in the `test` split. Additionally, the images required very little pre-processing:

- Pixel values were mapped to `[0,1]`.
- Images were converted into numpy arrays or tensors.
- In the case of transfer learning, pixel values were scaled to the region in which the pre-trained model was fit into ImageNet.

## How to Use This

The project folder contains the following files and directories:

```
asl_alphabet.ipynb      # Main notebook
assets/                 # Auxiliary images
requirements.txt        # Dependencies
utils.py                # Helper script: dataset loading
```

The main notebook [`asl_alphabet.ipynb`](asl_alphabet.ipynb) carries out all the research and almost everything is implemented there; the notebook uses [`utils.py`](utils.py) to load the dataset with the specifications mentioned in the previous section.

As the first step, the [dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) needs to be downloaded to the folder `./data/`, where [`utils.py`](utils.py) expects it.

Then, we have two options:

1. Clone this repository, install the [dependencies](#dependencies) and excute [`asl_alphabet.ipynb`](asl_alphabet.ipynb) locally.
2. Open [`asl_alphabet.ipynb`](asl_alphabet.ipynb) in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mxagar/asl_alphabet_image_classification/blob/main/asl_alphabet.ipynb)

For the last option, the local dataset can be uploaded to your GDrive, and from there, to the virtual machine instance of your Colab workspace. The steps are explained in the notebook.

### Dependencies

You should create a python environment (e.g., with [conda](https://docs.conda.io/en/latest/)) and install the dependencies listed in the [`requirements.txt`](requirements.txt) file; no specific versions are needed. A short summary of commands required to have all in place is the following:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Discussion on the Used Models

As mentioned in the introduction, this is an experimental (and on-going) project in which I try some Keras functionalities related to CNNs on an *easy* dataset. The used models and their accuracy metric on the test dataset are the following:

1. A from-scratch defined and trained CNN model: `0.998`.
2. A frozen VGG16 backbone with a fully connected classifier network: `0.149`.
3. An autoencoder from which the compressed representations are used to fit a random forest: `0.995`.

If our goal is to define and train the most accurate model, the first model wins. In the following, notes about the motivation, definition, training and evaluation of each model are provided. Additionally, [possible improvements](#next-steps-improvements) are listed at the end.

### CNN from Scratch

The model is rather simple: it consists of 4 convolution-maxpooling layers that increase the number of channels from `16` to `128` by reducing the activation map size down to `3 x 3`. The final classifier consists of two fully connected layers with dropout in-between to control overfitting. It resembles the good old [LeNet](https://en.wikipedia.org/wiki/LeNet).

The learning curves of the RMSProp optimization algorithm behaved nicely and the training stopped at epoch 12 due to early stopping on the validation split (20% of the `train` split):

![From Scratch CNN: Learning Curves](./assets/cnn_scratch_learning_curves.jpg)

The confusion matrix looks also very nice:

![From Scratch CNN: Confusion Matrix](./assets/confusion_matrix_scratch.jpg)

The following figure shows 16 of the 27 / 15000 missclassified images:

![From Scratch CNN: Missclassifications](./assets/cnn_scratch_missclassifications.jpg)

### Fine-Tuning and Transfer Learning of ResNet50 and VGG16

I tried two backbones or networks trained on [ImageNet](https://www.image-net.org):

- [ResNet50](https://en.wikipedia.org/wiki/Residual_neural_network)
- [VGG16](https://www.geeksforgeeks.org/vgg-16-cnn-model/)

And I applied two techniques

- Transfer learning, i.e., training of the appended classifier only, with weights of the pre-trained network frozen.
- Fine-tuning: complete training of the network, starting with the pre-trained weights.

My initial assumption was that ResNet50 with fine-tuning should be the best option, due to the specific image classes and the large dataset. However, it's the VGG16 with transfer learning the one that best performed &mdash; although the accuracy is very bad, compared to the other models: `0.149`.

In the following, the confusion matrix achieved with this approach (a complete mess :sweat_smile:):

![Transfer Learning: Confusion Matrix](./assets/confusion_matrix_transfer_learning.jpg)

This model type definitely needs a better analysis from my side to figure out what's going on.

### Autoencoder Compression + Random Forest

Autoencoders are able to compress data observations, i.e., images in the present case, to latent vectors. They achieve that with an encoder-decoder architecture which (1) reduces the dimensionality of the input sample to a bottleneck and then (2) expands it to obtain a reconstructed representation that is intended to be as close a possible to the input.

The used encoder architecture is very similar to the CNN model created from scratch; the decoder expands the latent vector with transpose convolutions. The following is an example of an original image and its reconstruction:

![Autoencoder: Reconstruction](./assets/autoencoder.jpg)

Once the autoencoder was trained, I used it to encode all the images to vectors of size `512`; then, I attached two classifiers:

- A logistic regression, to provide some intepretability to the model choices.
- A random forest, to be able to reach high accuracies.

Both models were trained with a small grid search using cross-correlation (i.e., hyperparameter tuning). Unfortunately, the logistic regression was not able to converge, and only the results of the best random forest are shown below, which are close to the CNN model defined and trained from scratch:

![Autoencoder + Random Forest: Confusion Matrix](./assets/confusion_matrix_autoencoder_rf.jpeg)

The following figure shows 16 of the 77 / 15000 missclassified images:

![Autoencoder: Missclassifications](./assets/autoencoder_missclassifications.jpg)

## Preliminary Conclusions

The preliminary tests indicate that the the CNN defined and trained from scratch outperforms the more complex models in terms of classification accuracy; the used architecture is similar to [LeNet](https://en.wikipedia.org/wiki/LeNet). That is probably because the dataset is very easy: small images with relatively few, well-defined and balanced classes.

It seems that an autoencoder which has an encoder with a similar architecture as the outperforming CNN is able to compress `~ 15x` the images preserving relevant information. At least, the random forest is able to achieve a similar performance using the encoded representations. Maybe that's an approach to introduce some intepretability to CNNs.

## Next Steps, Improvements

- [ ] Try gray images. Enough information should be there and we could speed up the training & the inference, and improve the performance metrics even more?
- [ ] Transfer learning/Fine-tuning: analyze why it's not working.
- [ ] Being a well balanced multiclass dataset, accuracy is not that bad of a choice; however, in general, we should consider other metrics:
  - Precision: to optimize/evaluate Type I error
  - Recall: to optimize/evaluate Type II error
  - F1: harmonic mean of precision & recall
  - [Matthews Correlation Coefficient (MCC)](https://towardsdatascience.com/the-best-classification-metric-youve-never-heard-of-the-matthews-correlation-coefficient-3bf50a2f3e9a)
- [ ] Try variational autoencoders for decoupling the compressed dimensions and better interpretability.
- [ ] Tray smaller compressed representations; where is the limit?
- [ ] Try interpreting the choices of the autoencoder-based model.
- [ ] Manifold learning of the compressed representations: T-SNE or similar.

<!--
## Interesting Links
-->

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

You are free to use this project, but please link it back to the original source.

<!--
## Requirements

- Submit report as a PDF.
- Deep learning model for any task we select, using the dataset of our choice.
- Describe the dataset.
- Explain main objectives: problem type, goals.
- Describe briefly: data exploration, cleaning, feature engineering.
- Variations of a deep learning model: at least 3; different hyperparameters, etc.
- Explain model recommendation: choose explainability / accuracy?
- Explain key findings.
- Next steps: issues, improvements, etc.
-->
