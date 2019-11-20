import sys
import random
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import ujson
import numpy as np

import matplotlib
import matplotlib.cm as cm
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D

latents = ujson.load(open('experiments/tsne_latents.json', 'r'))
latents = np.mean(np.array(latents), axis=1)

print (latents.shape)

labels = ujson.load(open('experiments/tsne_labels.json', 'r'))
persons = ujson.load(open('experiments/tsne_persons.json', 'r'))
print (np.array(persons).shape)
print (np.array(labels).shape)

#label_dict = {'M': '0', 'F' : '1'}
label_dict = {l:i for l,i in zip(set(labels), range(0,len(set(labels))))}
#label_dict = {'0': '0', '1' : '1', '2' : '2', '3' : '3', '4' : '4', '5' : '5', '6' : '6'}
print(label_dict)
# pers_dict = {'249':0, '239':1, '276':2, '283':3, '243':4, '254':5, '258':6, '271':7}
#pers_dict = {'4bf764f4f':0, '3ecbd1402':1, '5ba7890ab':2, '2debca276':3, '664ea0ccc':4, 'd1dae148e':5, '05eb0bb02_1':6}
pers_dict = {p:i for p,i in zip(set(persons), range(0,len(set(persons))))}
print(pers_dict)

colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', '0.75']

labels = [label_dict[p] for p in labels]
persons = [pers_dict[p] for p in persons]
print(persons)

labels_distribution = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0 }
for label in labels:
    labels_distribution[str(label)] += 1
print(labels_distribution)
persons_distribution = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0 }
for person in persons:
    persons_distribution[str(person)] += 1
print(persons_distribution)

labels = np.array(labels)

persons = np.array(persons)

print (latents.shape)
print (labels.shape)

#pca = PCA(n_components=10, whiten=False)
#latents = pca.fit(latents).transform(latents)
#tsne_x = np.array(latents[:,1])
#tsne_y = np.array(latents[:,2])

#latents = np.delete(latents, np.s_[7:256], 1)
#latents = np.delete(latents, np.s_[0:100], 1)
#pca = PCA(n_components=50, whiten=False)
#latents = pca.fit(latents).transform(latents)
#latents = np.delete(latents, np.s_[1:2], 1)

"""
for index in range(1, 256):
    print(index)
    temp = np.delete(latents, np.s_[index:256], 1)

    l_embedded = TSNE(n_components=2, n_iter=750).fit_transform(temp)
    tsne_x = np.array(l_embedded[:, 0])
    tsne_y = np.array(l_embedded[:, 1])

    plt.scatter(tsne_x, tsne_y, c=persons, cmap=matplotlib.colors.ListedColormap(colors), label=labels)
    plt.legend
    plt.title('Speaker ID')
    plt.xlabel('t-SNE Projected Latent')
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    pylab.savefig('experiments/tsne_speaker_id'+str(index)+'.png')

    plt.clf()

    # plt.scatter(tsne_x, tsne_y, c=labels, cmap=matplotlib.colors.ListedColormap(colors), label=labels)
    plt.scatter(tsne_x, tsne_y, c=labels, cmap=matplotlib.colors.ListedColormap(colors), label=labels)
    plt.legend
    plt.title('All seven emotions')
    plt.xlabel('t-SNE Projected Latent')
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    pylab.savefig('experiments/tsne_emotions'+str(index)+'.png')
    plt.clf()
"""
print (latents.shape)

l_embedded = TSNE(n_components=2, n_iter=750).fit_transform(latents)

tsne_x = np.array(l_embedded[:,0])
tsne_y = np.array(l_embedded[:,1])

print (tsne_x.shape)
print (tsne_y.shape)
print (persons.shape)

plt.scatter(tsne_x, tsne_y, c=persons, cmap=matplotlib.colors.ListedColormap(colors), label=labels)
plt.legend
plt.title('Speaker ID')
plt.xlabel('t-SNE Projected Latent')
ax = plt.gca()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
pylab.savefig('experiments/tsne_speaker_id.png')

plt.clf()

#plt.scatter(tsne_x, tsne_y, c=labels, cmap=matplotlib.colors.ListedColormap(colors), label=labels)
plt.scatter(tsne_x, tsne_y, c=labels, cmap=matplotlib.colors.ListedColormap(colors), label=labels)
plt.legend
plt.title('All seven emotions')
plt.xlabel('t-SNE Projected Latent')
ax = plt.gca()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
pylab.savefig('experiments/tsne_emotions.png')
