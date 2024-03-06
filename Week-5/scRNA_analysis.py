from sklearn.datasets import load_breast_cancer
import scanpy as sc

# Load the same breast cancer dataset as above from scikit-learn
data = load_breast_cancer()

# Create an AnnData object from the dataset
adata = sc.AnnData(data.data)
adata.obs['target'] = data.target

# Preprocessing
sc.pp.filter_genes(adata, min_counts=1)  # Filter genes expressed in at least 1 cell
sc.pp.normalize_total(adata)  # Normalize total counts per cell
sc.pp.log1p(adata)  # Log-transform the data
sc.pp.highly_variable_genes(adata, n_top_genes=2000)  # Identify highly variable genes
adata = adata[:, adata.var.highly_variable]  # Filter highly variable genes

# Quality control
sc.pp.pca(adata, n_comps=20)  # Perform PCA
sc.pl.pca_variance_ratio(adata, log=True)  # Plot PCA variance ratio

# Clustering
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)  # Compute neighborhood graph
sc.tl.leiden(adata)  # Cluster cells using the Leiden algorithm
sc.tl.umap(adata)  # Compute UMAP embedding

# Differential expression analysis
sc.tl.rank_genes_groups(adata, 'leiden')  # Find marker genes for each cluster

# Visualization
sc.pl.umap(adata, color=['leiden', 'target'])  # Plot UMAP with cluster and marker genes

# Save the analyzed data
adata.write('breast_cancer_analysis.h5ad')
