This repository contains the code and results of applying a multi-timescal Aux-VAE architecture on audio speach data, namely a slow auxiliary variable to extract a low dimensional representation for tasks that depend on features that does not change with time like speaker identification and gender classification, and another auxiliary variable for tasks that depend on features,which change in time such as emotion recognition. The model was implemented in [PyTorch](https://github.com/pytorch/pytorch).

### Multi-Timescale Aux-VAE
<p align="center"><img src="./imgs/Two-Aux-VAE.png" width="600" /></p>

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

To train a linear classifier, using the OMG-Emotion dataset, on top of representations learned by the model, run:
```
python speaker_id.py --pretrained-model=TYPE_OF_MODEL --task={0,1}
```
where "0" for speaker identification and "1" for gender classification.

To train a linear classifier, using the LibriSpeech dataset, on top of representations learned by the model, run:
```
python librispeech_speaker_id.py --pretrained-model=TYPE_OF_MODEL --task={0,1}
```
where "0" for speaker identification and "1" for gender classification.
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
