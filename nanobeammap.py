import numpy as np
import py4DSTEM
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import NMF
from umap import UMAP
from tqdm import tqdm
import gc

class NanoBeamMap:
    def __init__(self, filename):
        self.filename = filename
        self.dataset = py4DSTEM.import_file(filename)
        self.scan_shape = self.dataset.data.shape[:2]
        self.diffraction_shape = self.dataset.data.shape[2:]
        self.total_patterns = np.prod(self.scan_shape)
        self.flattened_shape = (self.total_patterns, np.prod(self.diffraction_shape))
        self.processed_data = None
        self.reduced_data = None
        self.labels = None
        self.cluster_centers = None
        self.probabilities = None
        self.average_patterns = None

    def preprocess(self, chunk_size=1000):
        print("Preprocessing data...")
        chunks = [(i, min(i+chunk_size, self.total_patterns)) 
                  for i in range(0, self.total_patterns, chunk_size)]
        
        memmap_filename = 'processed_data.dat'
        self.processed_data = np.memmap(memmap_filename, dtype='float32', mode='w+', shape=self.flattened_shape)
        
        scaler = StandardScaler()
        
        for start, end in tqdm(chunks, desc="Processing chunks"):
            chunk = self.dataset.data.reshape(self.flattened_shape)[start:end]
            chunk = chunk.astype('float32')
            
            np.log1p(chunk, out=chunk)
            scaler.partial_fit(chunk)
            self.processed_data[start:end] = chunk
        
        for start, end in tqdm(chunks, desc="Standardizing"):
            chunk = self.processed_data[start:end]
            self.processed_data[start:end] = scaler.transform(chunk)
        
        print("Preprocessing complete.")
        gc.collect()

    def reduce_dimensionality(self, n_components_pca=50, n_neighbors=15, min_dist=0.1, chunk_size=1000):
        print("Reducing dimensionality...")
        chunks = [(i, min(i+chunk_size, self.total_patterns)) 
                  for i in range(0, self.total_patterns, chunk_size)]
        
        ipca = IncrementalPCA(n_components=n_components_pca)
        for start, end in tqdm(chunks, desc="IPCA fitting"):
            ipca.partial_fit(self.processed_data[start:end])
        
        pca_result = np.empty((self.total_patterns, n_components_pca), dtype='float32')
        for start, end in tqdm(chunks, desc="IPCA transform"):
            pca_result[start:end] = ipca.transform(self.processed_data[start:end])
        
        umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=3, low_memory=True)
        self.reduced_data = umap.fit_transform(pca_result)
        
        print("Dimensionality reduction complete.")
        gc.collect()

    def segment_data(self, method='gmm', n_clusters=None, **kwargs):
        print(f"Segmenting data using {method.upper()}...")
        if method == 'gmm':
            self.fit_gmm(n_clusters, **kwargs)
        elif method == 'kmeans':
            self.fit_kmeans(n_clusters, **kwargs)
        elif method == 'dbscan':
            self.fit_dbscan(**kwargs)
        elif method == 'nmf':
            self.fit_nmf(n_clusters, **kwargs)
        else:
            raise ValueError("Unsupported segmentation method")
        print("Segmentation complete.")

    def fit_gmm(self, n_components, **kwargs):
        gmm = GaussianMixture(n_components=n_components, **kwargs)
        self.labels = gmm.fit_predict(self.reduced_data)
        self.cluster_centers = gmm.means_
        self.probabilities = gmm.predict_proba(self.reduced_data)

    def fit_kmeans(self, n_clusters, **kwargs):
        kmeans = KMeans(n_clusters=n_clusters, **kwargs)
        self.labels = kmeans.fit_predict(self.reduced_data)
        self.cluster_centers = kmeans.cluster_centers_

    def fit_dbscan(self, **kwargs):
        dbscan = DBSCAN(**kwargs)
        self.labels = dbscan.fit_predict(self.reduced_data)

    def fit_nmf(self, n_components, **kwargs):
        nmf = NMF(n_components=n_components, **kwargs)
        self.components = nmf.fit_transform(self.reduced_data)
        self.feature_components = nmf.components_

    def compute_average_diffraction_patterns(self):
        n_components = len(np.unique(self.labels))
        self.average_patterns = np.zeros((n_components, *self.diffraction_shape))
        for i in range(n_components):
            mask = self.labels == i
            self.average_patterns[i] = np.mean(self.dataset.data.reshape(self.total_patterns, -1)[mask], axis=0).reshape(self.diffraction_shape)

    def visualize_spatial_distribution(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.labels.reshape(self.scan_shape), cmap='viridis')
        plt.title('Spatial Distribution of Segments')
        plt.colorbar(label='Segment')
        plt.show()

    def visualize_patterns(self, patterns, titles, scale='linear'):
        n_components = patterns.shape[0]
        fig, axs = plt.subplots(1, n_components, figsize=(5*n_components, 5))
        if n_components == 1:
            axs = [axs]
        
        for i in range(n_components):
            pattern = patterns[i]
            
            if scale == 'sqrt':
                pattern = np.sqrt(pattern)
                scale_label = 'Sqrt'
            elif scale == 'log':
                pattern = np.log1p(pattern)
                scale_label = 'Log'
            else:
                scale_label = 'Linear'
            
            im = axs[i].imshow(pattern, cmap='viridis')
            axs[i].set_title(f'{titles[i]}\n({scale_label} scale)')
            fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()

    def visualize_average_patterns(self, scale='linear'):
        if self.average_patterns is None:
            self.compute_average_diffraction_patterns()
        
        titles = [f'Average Pattern {i+1}' for i in range(self.average_patterns.shape[0])]
        self.visualize_patterns(self.average_patterns, titles, scale)

    def visualize_cluster_centers(self, scale='linear'):
        if self.cluster_centers is None:
            print("Cluster centers not available.")
            return
        
        if self.cluster_centers.shape[1] != np.prod(self.diffraction_shape):
            print("Note: Cluster centers are in reduced dimension space.")
            self.visualize_patterns(self.cluster_centers, [f'Component {i+1}' for i in range(self.cluster_centers.shape[0])], scale='linear')
        else:
            centers = self.cluster_centers.reshape(-1, *self.diffraction_shape)
            titles = [f'Component {i+1}' for i in range(centers.shape[0])]
            self.visualize_patterns(centers, titles, scale)

    def visualize_nmf_components(self, scale='linear'):
        if self.feature_components is None:
            print("NMF components not available.")
            return
        
        components = self.feature_components.reshape(-1, *self.diffraction_shape)
        titles = [f'NMF Component {i+1}' for i in range(components.shape[0])]
        self.visualize_patterns(components, titles, scale)