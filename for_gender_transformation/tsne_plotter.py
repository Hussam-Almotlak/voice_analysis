import sys
import random
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from sklearn.manifold import TSNE
import ujson
import numpy as np

import matplotlib
import matplotlib.cm as cm
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D

globel_latent = ujson.load(open('experiments/tsne_globel_latent.json', 'r'))
globel_latent = np.mean(np.array(globel_latent), axis=1)

medium_latent = ujson.load(open('experiments/tsne_medium_latent.json', 'r'))
medium_latent = np.mean(np.array(medium_latent), axis=1)

print (globel_latent.shape)

labels = ujson.load(open('experiments/tsne_labels.json', 'r'))
persons = ujson.load(open('experiments/tsne_persons.json', 'r'))
print (np.array(persons).shape)
print (np.array(labels).shape)

label_dict = {l:i for l,i in zip(set(labels), range(0,len(set(labels))))}
print(label_dict)
pers_dict = {p:i for p,i in zip(set(persons), range(0,len(set(persons))))}
print(pers_dict)

colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black']
cdict = {0:'red', 1:'blue', 2:'green', 3:'cyan', 4:'magenta', 5:'yellow', 6:'black'}
emotions = {0:'Anger', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
label_group = np.array([0, 1, 2, 3, 4, 5, 6])

labels = np.array([label_dict[p] for p in labels])
persons = np.array([pers_dict[p] for p in persons])
print(persons)

labels_distribution = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0 }
for label in labels:
    labels_distribution[str(label)] += 1
print(labels_distribution)
persons_distribution = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0 }

labels = np.array(labels)
persons = np.array(persons)

print (globel_latent.shape)
print (labels.shape)

print (globel_latent.shape)

l_embedded_globel = TSNE(n_components=2, n_iter=750).fit_transform(globel_latent)
tsne_x = np.array(l_embedded_globel[:,0])
tsne_y = np.array(l_embedded_globel[:,1])

print (tsne_x.shape)
print (tsne_y.shape)
print (persons.shape)

plt.scatter(tsne_x, tsne_y, c=persons, cmap=matplotlib.colors.ListedColormap(colors), label=persons)
plt.legend
plt.title('Speaker ID')
plt.xlabel('t-SNE Projected Latent')
ax = plt.gca()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
pylab.savefig('experiments/tsne_speaker_id_global_latent.png')
plt.clf()

ig, ax = plt.subplots()
for label in label_group:
    index = []
    for i in range(0, len(labels)):
        if labels[i] == label:
            index.append(i)
    ax.scatter(tsne_x[index], tsne_y[index], c= cdict[label], label= emotions[label])
ax.legend()
pylab.savefig('experiments/tsne_emotions_global_latent.png')
plt.clf()


l_embedded_medium = TSNE(n_components=2, n_iter=750).fit_transform(medium_latent)
tsne_x_m = np.array(l_embedded_medium[:,0])
tsne_y_m = np.array(l_embedded_medium[:,1])

plt.scatter(tsne_x_m, tsne_y_m, c=persons, cmap=matplotlib.colors.ListedColormap(colors), label=persons)
plt.legend
plt.title('Speaker ID')
plt.xlabel('t-SNE Projected Latent')
ax = plt.gca()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
pylab.savefig('experiments/tsne_speaker_id_medium_latent.png')
plt.clf()

ig, ax = plt.subplots()
for label in label_group:
    index = []
    for i in range(0, len(labels)):
        if labels[i] == label:
            index.append(i)
    ax.scatter(tsne_x_m[index], tsne_y_m[index], c= cdict[label], label= emotions[label])
ax.legend()
pylab.savefig('experiments/tsne_emotions_medium_latent.png')
plt.clf()
