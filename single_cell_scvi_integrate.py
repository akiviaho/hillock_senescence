import scanpy as sc
import anndata as ad
import scib
import scvi
import seaborn as sns
import torch
import os
from matplotlib import pyplot as plt
from datetime import datetime

# Author: Antti Kiviaho
# Date: 12.12.2024
#
# A script for running scvi integration on single cell datasets
# on the Lyu et al. 2024 dataset
if __name__ == "__main__":

    # Load the data
    adata = sc.read_h5ad('./single-cell/lyu_2024/adata_obj_qc_normalized_orig.h5ad')
    adata.uns['log1p']["base"] = None

    # Preprocess and scale
    adata.obs['sample'] = adata.obs['sample'].astype('category')
    scib.preprocessing.scale_batch(adata,batch='sample')
    print('Scaling done...')

    adata = scib.preprocessing.hvg_batch(adata,batch_key='sample',target_genes=2000,flavor='seurat',adataOut=True)
    print('HVGs calculated...')

    print('CUDA is available: ' + str(torch.cuda.is_available()))
    print('GPUs available: ' + str(torch.cuda.device_count()))

    print('Initiating training on GPU ...')
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key='sample')
    vae = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")

    vae.train(use_gpu=True)

    adata.obsm["X_scVI"] = vae.get_latent_representation()

    sc.pp.neighbors(adata, use_rep="X_scVI",random_state=23523416)
    sc.tl.leiden(adata, key_added="VI_clusters",random_state=23523416)
    sc.tl.umap(adata,random_state=23523416)
    print('NN graph, UMAP & Leiden ready...')

    adata.write_h5ad('./single-cell/lyu_2024/adata_obj_qc_normalized_orig_scvi_integrated.h5ad')
    print('SCVI integration saved...')

