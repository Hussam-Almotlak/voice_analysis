This repository contains the code and results of applying a VAE architecture with two auxiliary variables on speach audio data, namely a slow auxiliary variable to extract a low dimensional representation for tasks that depend on features that does not change with time like speaker identification, and another auxiliary variable for tasks that depend on features that change in time like emotion recognition. The model was implemented in [PyTorch](https://github.com/pytorch/pytorch).


## Running the Code

### Preprocessing

In order to preprocess the [OMG dataset](https://github.com/knowledgetechnologyuhh/OMGEmotionChallenge), please download the audio version of the dataset and then run:
```
python prepare_OMG.py --OMG-path={DIR TO OMG AUDIOS}
```

{DIR TO VCTK DIRECTORY} to be replaced by the directory to the audio data

### Running the Model

Models can be trained by running the main.py script. Important options include batch-size, learning-rate, the size of the hidden representaion and number of epochs

The Aux-VAE model can be trained by:
```
python main.py --model-type=vae_g_l
```

For additional options (such as setting hyperparameters) see:
```
python main.py --h
```

To analyse latent representations first run:
```
python main.py --model-type=TYPE_OF_MODEL --mode=analyse-latent
```
To create t-SNE plots then run:
```
python tsne_plotter.py
```

To train a linear classifier on top of representations learned by a model in order to perform speaker identification run:
```
python speaker_id.py --pretrained-model=TYPE_OF_MODEL
```

### Dependencies
* [Numpy](http://www.numpy.org)
* [Scipy](https://www.scipy.org)
* [LibROSA](https://librosa.github.io/librosa/)
* [SOX](http://sox.sourceforge.net)
* [PyTorch](https://pytorch.org)
* [torchaudio](https://github.com/pytorch/audio)
* [ujson](https://pypi.org/project/ujson/)
* [tqdm](https://github.com/tqdm/tqdm)
* [tensorboardX](https://github.com/lanpa/tensorboardX)
* [Scikit-Learn](http://scikit-learn.org/stable/) (optional, for tsne plots and silhouette scores)
