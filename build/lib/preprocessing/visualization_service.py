import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import logging

log = logging.getLogger(__name__)

class Visualization:
    def __init__(self,
                 data: np.ndarray,
                 title: str = None,
                 xlabel_index: int = 0,
                 ylabel_index: int = 1,
                 zlabel_index: int = None,
                 x_label_title: str = "x",
                 y_label_title: str = "y",
                 z_label_title: str = "z",
                 save_path:str=None
                 ):
        self.data = data
        self.title = title
        self.xlabel_index = xlabel_index
        self.ylabel_index = ylabel_index
        self.zlabel_index = zlabel_index
        self.xlabel = x_label_title
        self.ylabel = y_label_title
        self.zlabel = z_label_title
        self.save_path=save_path

    def plot_2d(self, clusters: np.ndarray):
        """
        Plots the data in 2D, color-coded by cluster labels.

        Parameters:
            clusters (np.ndarray): An array of cluster labels.
        """
        # Create a scatter plot
        # Flatten the clusters to get labels
        labels = np.zeros(len(self.data), dtype=int) - 1
        for cluster_id, cluster in enumerate(clusters):
            for index in cluster:
                labels[index] = cluster_id

        # Plot the clustered data
        plt.figure(figsize=(10, 6))

        # Plot all PCA data points in light grey for context
        plt.scatter(self.data[:, 0], self.data[:, 1], color='lightgrey', alpha=0.5)
        mask = labels >= 0
        sns.scatterplot(x=self.data[mask,self.xlabel_index], y=self.data[mask,self.ylabel_index], hue=labels[mask], palette='viridis', s=100, alpha=0.7)
        # Add labels and title
        plt.xlabel(f"Feature {self.xlabel}")
        plt.ylabel(f"Feature {self.ylabel}")
        plt.title(self.title)
        # Show the plot
        plt.show()
        if self.save_path:
            plt.savefig(save_path,dpi=300, bbox_inches='tight')

    def plot_3d(self, clusters: np.ndarray):
        """
        Plots the data in 3D, color-coded by cluster labels.

        Parameters:
            labels (np.ndarray): An array of cluster labels.
            :param clusters:
        """


        # Flatten the clusters to get labels
        labels = np.zeros(len(self.data), dtype=int) - 1  # Initialize with -1 for non-clustered points
        for cluster_id, cluster in enumerate(clusters):
            for index in cluster:
                labels[index] = cluster_id
        # Plot the clustered data in 3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all PCA data points in light grey for context
        ax.scatter(self.data[:, self.xlabel_index], self.data[:, self.ylabel_index], self.data[:, self.zlabel_index], color='lightgrey', alpha=0.5)
        mask = labels >= 0
        # Create a 3D scatter plot
        sc = ax.scatter(self.data[mask, self.xlabel_index], self.data[mask, self.ylabel_index], self.data[mask, self.zlabel_index], c=labels[mask], cmap='viridis', s=100,
                        alpha=0.7)


        # Add labels and title
        ax.set_xlabel(f"Feature {self.xlabel}")
        ax.set_ylabel(f"Feature {self.ylabel}")
        ax.set_zlabel(f"Feature {self.zlabel}")
        plt.title(self.title)
        # Show the plot
        plt.show()
        if self.save_path:
            plt.savefig(save_path,dpi=300, bbox_inches='tight')