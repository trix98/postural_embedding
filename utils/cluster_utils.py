import os
import numpy as np
import pickle
from glob import glob
import logging
import hdbscan
from sklearn.model_selection import GridSearchCV
import hdbscan
from sklearn.metrics import make_scorer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import random

random.seed(0)


def manage_dir(out_dir, base_dir="./data/"):
    """if dir does not exist make dir and return path as str"""
    out_dir = base_dir + out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir += "/"
    return out_dir


def get_features(feat_path):
    """open transfer learning features and store in numpy array"""
    passed_features = []
    n_files = len(glob(feat_path + '*' + '.p'))
    for i in range(n_files):
        with open(feat_path + "%d.p" % i, "rb") as file:
            passed_features.append(pickle.load(file))
    return np.array(passed_features)


def hdbscan_grid_search(embedding):
    """grid search hdbscan parameters, optimized by DBCV index"""
    logging.captureWarnings(True)
    hdb = hdbscan.HDBSCAN(gen_min_span_tree=True).fit(embedding)

    # specify parameters and distributions to sample from
    param_dist = {'min_samples': [2, 3, 5, 10, 20, 30, 50, 80, 100],
                  'min_cluster_size': [20, 30, 40, 60, 80, 100, 300, 600],
                  'cluster_selection_method': ['eom', 'leaf'],
                  'metric': ['euclidean', 'manhattan']
                  }

    validity_scorer = make_scorer(hdbscan.validity.validity_index, greater_is_better=True)
    grid_search = GridSearchCV(hdb, param_dist, scoring=validity_scorer, cv=5)
    grid_search.fit(embedding)
    info = f"Best Parameters {grid_search.best_params_} \n DBCV score :{grid_search.best_estimator_.relative_validity_}"
    print(info)
    return info


def cluster_counts(model, verbose=1):
    """create df of number of frames for each cluster"""
    labels = model.labels_
    cnts = pd.DataFrame(labels)[0].value_counts()
    cnts = cnts.reset_index()
    cnts.columns = ['cluster', 'count']
    pd.set_option('display.max_rows', None)
    if verbose:
        print("#clusters: ", labels.max())
        print(cnts.sort_values(['count'], ascending=0))
    return cnts


def plots_2d_umap(features):
    """
    visualise the 2D UMAP embedding space
    in the scatter plot points are coloured by time index
    """
    emb = umap.UMAP(n_components=2, random_state=0).fit(features)
    plt.figure()
    plt.scatter(emb.embedding_[:, 0], emb.embedding_[:, 1], s=4,
                c=np.arange(2501))  # , c=y_train, cmap='Spectral')
    #  plt.title('Embedding of the training set by UMAP', fontsize=24);
    plt.xlabel('Embedding 1', weight='bold')
    plt.ylabel('Embedding 2', weight='bold')
    plt.savefig(manage_dir('results') + '2D_UMAP_Scatter' + '.png', dpi=300, bbox_inches="tight")

    plt.figure()
    kdeplot = sns.jointplot(x=emb.embedding_[:, 0], y=emb.embedding_[:, 1], kind="kde",
                            cbar=True, shade=True, cmap="Blues")
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    pos_joint_ax = kdeplot.ax_joint.get_position()
    pos_marg_x_ax = kdeplot.ax_marg_x.get_position()
    kdeplot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    kdeplot.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
    kdeplot.set_axis_labels('x', 'y')
    kdeplot.ax_joint.set_xlabel('Embedding 1', weight='bold')
    kdeplot.ax_joint.set_ylabel('Embedding 2', weight='bold')

    plt.savefig(manage_dir('results') + '2D_UMAP_kde' + '.png', dpi=300, bbox_inches="tight")
    # plt.show()


def plots_cluster(counts, model, embedding):
    """
    bar chart of number of counts for each cluster
    & cluster assignment for the first 2D of the X UMAP embeddings
    :param counts: pd df of counts for each cluster (run cluster_counts())
    :param model: hdbscan cluster model
    :param embedding: UMAP embeddings
    """
    plt.figure()
    sns.barplot(data=counts, x='cluster', y='count', order=counts['cluster'], palette='viridis')
    plt.xlabel('Cluster', weight='bold')
    plt.ylabel('Count', weight='bold')
    plt.savefig(manage_dir('results') + 'bar_chart_clustering' + '.png', dpi=300, bbox_inches="tight")

    plt.figure()
    color_palette = sns.color_palette('Paired', 30)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in model.labels_]
    plt.scatter(x=embedding[:, 0], y=embedding[:, 1], c=cluster_colors, s=4)
    plt.xlabel('Embedding 1', weight='bold')
    plt.ylabel('Embedding 2', weight='bold')
    plt.savefig(manage_dir('results') + 'Clustering in 30D, first 2D' + '.png', dpi=300, bbox_inches="tight")


def cluster_samples(model, raw_img_dir='./data/images/', rand=1):
    """
    PLot up to 100 samples from each cluster
    :param model: hdbscan cluster model
    :param raw_img_dir: dir of img folder (frames must be labelled 0, 1, 2, ..., n)
    :param rand: random frames belonging to cluster, recommended if >100 in a cluster
    """
    result_dir = manage_dir('results')
    clust_rng = range(-1, model.labels_.max() + 1)
    label_dic = dict.fromkeys(clust_rng)
    pred = list(model.labels_)
    for i, l in enumerate(pred):
        if label_dic[l] is None:
            label_dic[l] = [i]
        else:
            label_dic[l].append(i)

    cluster = list(clust_rng)

    for clust in cluster:
        if rand:
            random.shuffle(label_dic[clust])
        plt.figure(figsize=(16, 16), facecolor='white')

        for i, img_path in enumerate(label_dic[clust][:100]):
            img_name = str(img_path) + '.png'
            plt.subplot(10, 10, i + 1)
            img = plt.imread(raw_img_dir + img_name)  # change for raw
            plt.imshow(img)
            plt.title(img_name[:-4])  # title is frame index
            plt.axis("off")
        plt.savefig(manage_dir('cluster_samples', result_dir) + '{}.png'.format(str(clust)))
