import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import wandb
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

from embeddings.ntu_classes import CLASS_LABELS

CLASS_DICT = {k: v for k, v in enumerate(CLASS_LABELS)}

class EmbeddingPlotCallback:
    # TODO: Add settings to config
    def __init__(self, embeddings_per_batch=50, epoch_plot_interval=10, selected_classes=None):
        self.embeddings_per_batch = embeddings_per_batch
        self.epoch_plot_interval = epoch_plot_interval
        self.embeddings = {'query': [], 'key': []}
        self.motion_embeddings = {'query': [], 'key': []}
        self.bone_embeddings = {'query': [], 'key': []}
        self.embedding_labels = []
        # TODO: Make this cleaner, maybe we don't even need different sampling but shouldn't matter too much
        self.current_indices = []
        self.random_sample = False
        self.selected_classes = selected_classes

    def clean_storage(self):
        self.embeddings = {'query': [], 'key': []}
        self.motion_embeddings = {'query': [], 'key': []}
        self.bone_embeddings = {'query': [], 'key': []}
        self.embedding_labels = []
        self.current_indices = []
        self.random_sample = False

    # TODO: Remove duplicate code
    def store_embeddings(self, outputs):
        q_embeddings = outputs.get('query', None)
        k_embeddings = outputs.get('key', None)

        if q_embeddings is not None:
            q_embeddings = q_embeddings.detach().cpu().numpy()
        if k_embeddings is not None:
            k_embeddings = k_embeddings.detach().cpu().numpy()

        if q_embeddings is not None and q_embeddings.shape[0] > self.embeddings_per_batch:
            if self.random_sample:
                q_embeddings = q_embeddings[self.current_indices]
                if k_embeddings is not None:
                    k_embeddings = k_embeddings[self.current_indices]
            else:
                indices = np.random.choice(q_embeddings.shape[0], self.embeddings_per_batch, replace=False)
                self.random_sample = True
                self.current_indices = indices # Need to save indices to store the same labels later
                q_embeddings = q_embeddings[indices]
                if k_embeddings is not None:
                    k_embeddings = k_embeddings[indices]
        else:
            self.random_sample = False

        self.embeddings['query'].append(q_embeddings)
        if k_embeddings is not None:
            self.embeddings['key'].append(k_embeddings)

    def store_motion_embeddings(self, outputs):
        q_embeddings = outputs.get('query', None)
        k_embeddings = outputs.get('key', None)

        if q_embeddings is not None:
            q_embeddings = q_embeddings.detach().cpu().numpy()
        if k_embeddings is not None:
            k_embeddings = k_embeddings.detach().cpu().numpy()

        if q_embeddings is not None and q_embeddings.shape[0] > self.embeddings_per_batch:
            if self.random_sample:
                q_embeddings = q_embeddings[self.current_indices]
                if k_embeddings is not None:
                    k_embeddings = k_embeddings[self.current_indices]
            else:
                indices = np.random.choice(q_embeddings.shape[0], self.embeddings_per_batch, replace=False)
                self.random_sample = True
                self.current_indices = indices # Need to save indices to store the same labels later
                q_embeddings = q_embeddings[indices]
                if k_embeddings is not None:
                    k_embeddings = k_embeddings[indices]
        else:
            self.random_sample = False

        self.motion_embeddings['query'].append(q_embeddings)
        if k_embeddings is not None:
            self.motion_embeddings['key'].append(k_embeddings)

    def store_bone_embeddings(self, outputs):
        q_embeddings = outputs.get('query', None)
        k_embeddings = outputs.get('key', None)

        if q_embeddings is not None:
            q_embeddings = q_embeddings.detach().cpu().numpy()
        if k_embeddings is not None:
            k_embeddings = k_embeddings.detach().cpu().numpy()

        if q_embeddings is not None and q_embeddings.shape[0] > self.embeddings_per_batch:
            if self.random_sample:
                q_embeddings = q_embeddings[self.current_indices]
                if k_embeddings is not None:
                    k_embeddings = k_embeddings[self.current_indices]
            else:
                indices = np.random.choice(q_embeddings.shape[0], self.embeddings_per_batch, replace=False)
                self.random_sample = True
                self.current_indices = indices # Need to save indices to store the same labels later
                q_embeddings = q_embeddings[indices]
                if k_embeddings is not None:
                    k_embeddings = k_embeddings[indices]
        else:
            self.random_sample = False

        self.bone_embeddings['query'].append(q_embeddings)
        if k_embeddings is not None:
            self.bone_embeddings['key'].append(k_embeddings)

    def store_labels(self, labels):
        embedding_labels = labels
        embedding_labels = embedding_labels.detach().cpu().numpy()
        
        if self.random_sample:
            embedding_labels = embedding_labels[self.current_indices]

        self.embedding_labels.append(embedding_labels)

    def plot_tsne(self, epoch, logger, prefix="val", stage="", view='Joint', log=True, type='All'):
        if view == 'Joint':
            q_embeddings = np.concatenate(self.embeddings['query'])
            k_embeddings = np.concatenate(self.embeddings['key']) if self.embeddings['key'] else None
        elif view == 'Motion':
            q_embeddings = np.concatenate(self.motion_embeddings['query'])
            k_embeddings = np.concatenate(self.motion_embeddings['key']) if self.motion_embeddings['key'] else None
        elif view == 'Bone':
            q_embeddings = np.concatenate(self.bone_embeddings['query'])
            k_embeddings = np.concatenate(self.bone_embeddings['key']) if self.bone_embeddings['key'] else None
        embedding_labels = np.concatenate(self.embedding_labels)
        embedding_labels = np.concatenate((embedding_labels, embedding_labels), axis=0) if k_embeddings is not None else embedding_labels

        embeddings = np.concatenate((q_embeddings, k_embeddings), axis=0) if k_embeddings is not None else q_embeddings

        self.plot_embedding_visualization(logger, embeddings, embedding_labels, prefix=prefix, step=epoch, stage=stage, view=view, log=log, type=type)

    def plot_embedding_visualization(self, logger, samples, labels, prefix="val", clustering="TSNE", step=0, suffix="", stage="", view='Joint', log=True, type='All'):
        warnings.filterwarnings("ignore", category=UserWarning)
        """samples is a list of embeddings from each batch"""
        n_labels = len(CLASS_DICT.keys())

        x = samples
        y = labels
        
        if self.selected_classes is not None:
            selected_indices = np.isin(y, self.selected_classes)
            x = x[selected_indices]
            y = y[selected_indices]

        if clustering == "PCA":
            pca = PCA(n_components=2)
            res = pca.fit_transform(x)
            var = pca.explained_variance_ratio_[:2]
        elif clustering == "TSNE":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tsne = TSNE(n_components=2, init='pca', metric='cosine', random_state=42)
                res = tsne.fit_transform(x)
        else:
            raise Exception("Invalid clustering type. Should be PCA or TSNE")

        classes = np.expand_dims(y, axis=1)
        classes[classes >= n_labels] = n_labels
        class_names = np.vectorize(CLASS_DICT.get)(classes)

        if res.shape[0] != class_names.shape[0]:
            print(f"Error during plotting! res {res.shape} and class {class_names.shape} shapes not equal")
            return
        
        num_samples = len(class_names)
        stream_split = num_samples // 2 # Half the samples of each skeleton are query, the other half are key

        stream_array = np.concatenate([np.array(['query'] * stream_split + ['key'] * stream_split)])
        stream_array = np.expand_dims(stream_array, axis=1)
        
        df = pd.DataFrame(data=np.concatenate([res, class_names, stream_array], axis=1), columns=["Dimension 1", "Dimension 2", "Class", "Stream"])
        #df = pd.DataFrame(data=np.concatenate([res, class_names], axis=1),
                       # columns=["Dimension 1", "Dimension 2", "Class"])
        df = df.astype({"Dimension 1": np.float64, "Dimension 2": np.float64})
        
        if type == 'All' or type == 'Class':
            # Plot by Class
            plot_by(df, 'Class', prefix, clustering, step, stage, 'Class', logger, view, log)

        # if type == 'All' or type == 'Stream':
            # Plot by Stream
            # plot_by(df, 'Stream', prefix, clustering, step, stage, 'Stream', logger, view, log)

def plot_by(df, hue, prefix, clustering, step, stage, title_suffix, logger, view='Joint', log=True):
    g = sns.jointplot(data=df, x="Dimension 1", y="Dimension 2",
                      hue=hue, height=12, palette='nipy_spectral',
                      legend=False)
    g.ax_joint.set_xlim(df['Dimension 1'].min() - 1, df['Dimension 1'].max() + 1)
    g.ax_joint.set_ylim(df['Dimension 2'].min() - 1, df['Dimension 2'].max() + 1)

    g.figure.suptitle(f'{prefix} {view} Embedding {clustering} Plot by {title_suffix} - Epoch {step}')

    if log:
        plt.close(g.fig) 
    else:
        plt.show(g.fig)

    if not log:
        # Save figure
        save_directory = "embeddings/img"
        figure_filename = f"{save_directory}/{prefix}_{title_suffix}_{view}_embedding_{step}.png"
        # g.figure.savefig(figure_filename)

    # Logging the figure according to the stage
    if logger is not None:
        if stage == "train" and log:
            logger.log({f"TSNE {view} Embeddings Train by {title_suffix}": g.figure})
        elif stage == "val" and log:
            logger.log({f"TSNE {view} Embeddings Val by {title_suffix}": g.figure})
