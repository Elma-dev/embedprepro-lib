import click
import pandas as pd
import numpy as np
from . import embedding_service
from . import agglomerative_clustering
from . import dimensionality_reduction
from . import visualization_service


@click.group()
def cli():
    pass


@click.argument("output_file", type=click.Path(), default="embbedding.npy")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--et", type=str, default="sentence_transformer", help="The type of embedder to use.")
@click.option("--mn", type=str, default="all-MiniLM-L6-v2", help="The model name to use.")
@click.option("--col", type=str, default="text", help="The column name to use.")
@click.option("--bs", type=int, default=32, help="The batch size to use.")
@click.option("--p", type=int, default=1, help="The number of parallel processes to use.")
@cli.command()
def embedding(input_file, output_file, et, mn, col, bs, p):
    data = pd.read_csv(input_file)[col]
    embeddingService = embedding_service.VectorEmbedService(
        model_name=mn,
        embedder_type=et)
    result = embeddingService.embed(
        sentences=data,
        batch_size=bs,
        parallel=p)
    np.save(output_file, result)


@click.argument("output_file", type=click.Path(), default="clusters.npy")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--et", type=str, default="sentence_transformer", help="The type of embedder to use.")
@click.option("--mn", type=str, default="all-MiniLM-L6-v2", help="The model name to use.")
@click.option("--col", type=str, default="text", help="The column name to use.")
@click.option("--bs", type=int, default=32, help="The batch size to use.")
@click.option("--p", type=int, default=1, help="The number of parallel processes to use.")
@click.option("--threshold", type=float, default=0.5, help="The threshold to use.")
@click.option("--min_cluster_size", type=int, default=1, help="The minimum cluster size to use.")
@click.option("--show_progress_bar", type=bool, default=True, help="Whether to show the progress bar.")
@click.option("--with_embedding", type=int, default=False, help="is embedding or not.")
@cli.command()
def clustring(
        input_file, output_file, et, mn, col, bs, p, threshold, min_cluster_size, show_progress_bar, with_embedding):
    agglomerativeClustering = agglomerative_clustering.AgglomerativeClustering(
        embedding_service=embedding_service.VectorEmbedService(et, mn))
    if with_embedding:
        data = np.load(input_file)
        clusters = agglomerativeClustering.cluster_with_embeddings(
            embeddings=data,
            threshold=threshold,
            min_cluster_size=min_cluster_size,
            batch_size=bs,
            show_progress_bar=show_progress_bar)
    else:
        data = pd.read_csv(input_file)[col]
        clusters = agglomerativeClustering.cluster(
            sentences=data,
            threshold=threshold,
            min_cluster_size=min_cluster_size,
            batch_size=bs,
            parallel=p,
            show_progress_bar=show_progress_bar)
    np.save(output_file, np.array(clusters, dtype=object))


@click.argument("output_file", type=click.Path(), default="dimreduction.npy")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--nc", type=int, default=2, help="The number of components to reduce to.")
@click.option("--ng", type=int, default=15, help="The number of neighbors to use.")
@click.option("--md", type=float, default=0.5, help="The minimum distance to use.")
@click.option("--metric", type=str, default="euclidean", help="The metric to use.")
@click.option("--et", type=str, default="sentence_transformer", help="The type of embedder to use.")
@click.option("--mn", type=str, default="all-MiniLM-L6-v2", help="The model name to use.")
@click.option("--col", type=str, default="text", help="The column name to use.")
@click.option("--algorithm", type=str, default="PCA", help="The algorithm to use.")
@click.option("--with_embedding", type=int, default=False, help="is embedding or not.")
@cli.command()
def reduction(
        input_file, output_file, nc, ng, md, metric, et, mn, col, algorithm, with_embedding):
    dimensionalityReduction = dimensionality_reduction.DimensionalityReduction(
        embedding_service=embedding_service.VectorEmbedService(et, mn),
        algorithm=algorithm, num_components=nc, num_neighbors=ng, metric=metric, min_dist=md)
    if with_embedding:
        data = np.load(input_file)
        new_dim = dimensionalityReduction.reduce_with_embeddings(
            embeddings=data)
    else:
        data = pd.read_csv(input_file)[col]
        new_dim = dimensionalityReduction.reduce(
            sentences=data)
    print(new_dim)
    np.save(output_file, new_dim)

@click.argument("reduced_data", type=click.Path(exists=True))
@click.argument("clusters_data", type=click.Path(exists=True))
@click.option("--xi", type=int, default=0, help="The index of the first dimension.")
@click.option("--yi", type=int, default=1, help="The index of the second dimension.")
@click.option("--zi", type=int, default=-1, help="The index of the third dimension.")
@click.option("--title", type=str, default="Clusters", help="The title of the plot.")
@click.option("--xlabel", type=str, default="X", help="The label of the x-axis.")
@click.option("--ylabel", type=str, default="Y", help="The label of the y-axis.")
@click.option("--zlabel", type=str, default="Z", help="The label of the z-axis.")
@click.option("--save", type=str, default=None, help="Save path")
@cli.command()
def visualization(
         reduced_data, clusters_data, xi, yi, zi, title, xlabel, ylabel, zlabel,save):
    visualizer=visualization_service.Visualization(
        data=np.load(reduced_data),
        title=title,
        xlabel_index=xi,
        ylabel_index=yi,
        zlabel_index=zi,
        x_label_title=xlabel,
        y_label_title=ylabel,
        z_label_title=zlabel,
        save_path=save
    )
    if zi==-1:
        visualizer.plot_2d(clusters=np.load(clusters_data, allow_pickle=True))
    else:
        visualizer.plot_3d(clusters=np.load(clusters_data,allow_pickle=True))


if __name__ == "__main__":
    cli()
