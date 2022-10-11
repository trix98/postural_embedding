
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap



def plots_2d_umap(features):
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
