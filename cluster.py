from utils.cluster_utils import *
import umap
import hdbscan

seed = 0
tune = 0

if __name__ == '__main__':
    feat_path = "./data/features/vgg19_fc2/"
    features = get_features(feat_path)  # load transfer learning extracted features

    if tune:
        for i in [2, 4, 6, 10, 15, 30, 50]:  # number of UMAP components queried
            initial_trans = umap.UMAP(n_components=i, random_state=seed).fit(features)
            print("=" * 20)
            print(str(i) + " components")
            hdbscan_grid_search(initial_trans.embedding_)

    # Here we have inputted the optimum number of components defined by > DBCV score
    trans = umap.UMAP(n_components=30, random_state=seed).fit(features)
    # print(trans, trans.embedding_.shape)

    # Here we have inputed the optimum parameters from the grid search
    model = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                            gen_min_span_tree=True, leaf_size=40, cluster_selection_method='eom',
                            metric='euclidean', min_cluster_size=20, min_samples=2, p=None)
    model.fit(trans.embedding_)

    # plot figures
    counts = cluster_counts(model, verbose=1)
    plots_2d_umap(features)
    plots_cluster(counts=counts, model=model, embedding=trans.embedding_)
    cluster_samples(model=model)
