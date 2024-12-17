import scanpy as sc
import scib
import anndata as ad
import os
import numpy as np
import pandas as pd
from tqdm import tqdm 
import seaborn as sns
from matplotlib import pyplot as plt
os.chdir('/lustre/scratch/kiviaho/hillock_club_senescence')


####################################### Lyu et al. 2024 data formatting #######################################

data_path = './single-cell/lyu_2024/raw_data'
sample_sheet = pd.read_csv('./single-cell/lyu_2024/lyu_2024_sample_sheet.txt',sep='\t')

adata_dict = {}
for i,row in tqdm(sample_sheet.iterrows(),unit='sample',total=len(sample_sheet)):
    geo_id = row['GEO_ID']
    sample_file = sorted([s for s in os.listdir(data_path) if geo_id in s])[0] # Take the first instance
    sample_prefix = ('_').join(sample_file.split('_')[:-1]) +'_' # Remove the file annotation
    dat = sc.read_10x_mtx(data_path, prefix = sample_prefix)
    obs_data = pd.DataFrame({
    'sample': np.repeat(row['Sample'].replace(' ','_'),len(dat)),
    'patient': np.repeat(row['Patient'].replace(' ','_'),len(dat)),
    'phenotype': np.repeat(row['Phenotype'].replace(' ','_'),len(dat))
    }, 
    index = dat.obs_names ) # Format obs data
    dat.obs = obs_data # Add the obs data into the data object
    adata_dict[row['Sample'].replace(' ','_')] = dat # Add the obs data into the data dict

# Add sample identifiers on the obs names
for k in adata_dict:
    dat = adata_dict[k]
    dat.obs_names = k + '_' + dat.obs_names

adata = ad.concat(adata_dict,join='outer') # Concatenate all the samples

# Save the data as a raw object
adata.write_h5ad('./single-cell/lyu_2024/adata_obj.h5ad')

####################################### Data processing as described in Lyu et al. 2024 #######################################

adata = sc.read_h5ad('./single-cell/lyu_2024/adata_obj.h5ad') # 236,856 cells × 33,538 genes

adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata,qc_vars=['mt'],inplace=True)

# Retain cells with less than 20% mitochondrial genes
adata = adata[adata.obs['pct_counts_mt'] < 20]

print(f'Size post MT% filtering: {adata.shape}') # 164,998 cells x 33,538 genes

# Step 2: Min number of detected genes 300, min UMIs 600 (no CellBender upstream)
sc.pp.filter_cells(adata,min_genes=300)
sc.pp.filter_cells(adata,min_counts=600)

# Step 3: Keep genes expressed in at least ten cells
sc.pp.filter_genes(adata, min_cells=10) 

print(f'Size post cell & gene count filtering: {adata.shape}') # 154,700 cells x 25,198 genes

adata = adata[~((adata[:, 'PF4'].X > 0).toarray().flatten())]
adata = adata[~((adata[:, 'HBB'].X > 1).toarray().flatten())]

print(f'Size post platelet & RBC filtering: {adata.shape}') # 152,476 cells x 25,198 genes

sc.external.pp.scrublet(adata)
adata = adata[adata.obs['predicted_doublet']==False]

print(f'Size post QC: {adata.shape}') # 152,476 cells x 25,198 genes --> No cells predicted as doublets here

# Save the raw counts
adata.layers['counts'] = adata.X.copy()

# Log2-transform the gene-expression matrix with the addition of 1 and normalize to 10,000 counts per cell
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Save the normalized log1p counts before scaling
adata.layers['log1p'] = adata.X.copy()

# Step 6: Perform highly variable gene selection using default parameters with the SCANPY function
sc.pp.highly_variable_genes(adata)

# Regress out UMI & MT% outliers --> Not in best practices, exclude the analysis step
#sc.pp.regress_out(adata, ['total_counts'])
#sc.pp.regress_out(adata, ['pct_counts_mt'])

# Scale to uniform mean & variance
sc.pp.scale(adata)

# Perform PCA on the dataframe
sc.pp.pca(adata,n_comps=50)

# Correct components using Harmony
sc.external.pp.harmony_integrate(adata,key='sample')

# Compute the neighborhood graph
sc.pp.neighbors(adata, use_rep='X_pca_harmony', random_state=519552572)

# Perform clustering
sc.tl.leiden(adata, key_added='leiden', resolution=1, random_state=519552572)

# Compute UMAP
sc.tl.umap(adata, random_state=519552572)

# Plot UMAPs of 
sc.set_figure_params(figsize=(4, 4))
sc.pl.umap(adata, color=['leiden', 'phenotype', 'sample'], show=False, save=True)

adata.write_h5ad('./single-cell/lyu_2024/adata_obj_qc_normalized_harmony_orig.h5ad')

#################### Annotate clusters ####################

# Define marker genes
def keep_specified_markers(marker_list, keep_vals):
    new_marker_list = {}
    for k, v in marker_list.items():
        new_marker_list[k] = [val for val in v if val in keep_vals]
    return new_marker_list

refined_markers = {
    'Epithelial': ['EPCAM', 'KRT18', 'KRT8', 'KLK3', 'AR', 'MSMB', 'KRT5', 'KRT15', 'TP63', 'KRT7', 'KRT19', 'KRT4'],
    'Endothelial': ['VWF', 'SELE', 'FLT1', 'ENG'],
    'Fibroblast': ['LUM', 'DCN', 'IGF1', 'APOD', 'STC1', 'FBLN1', 'COL1A2', 'C7', 'IGFBP5', 'ACTA2'],
    'SMC': ['RGS5', 'ACTA2', 'TAGLN', 'BGN', 'MYL9', 'MYLK', 'CALD1'],
    'Mast': ['KIT', 'TPSB2', 'TPSAB1', 'CPA3', 'VWA5A', 'IL1RL1', 'CTSG', 'SLC18A2', 'ACSL4', 'MS4A2', 'GATA2'],
    'T cell': ['CD3D', 'CD3E', 'CD3G', 'CD8A', 'CD8B', 'IL7R', 'NKG7', 'CD7', 'GNLY'],
    'B cell': ['CD79A', 'MS4A1', 'CD79B', 'IGHM', 'CD83'],
    'Myeloid': ['C1QA', 'C1QB', 'C1QC', 'CD68', 'LYZ', 'IL1A', 'IL1B', 'S100A9', 'S100A8', 'CXCL8', 'FCGR3A', 'CSF1R'],
    'Neuronal': ['PLP1', 'MPZ', 'MT1H'],
    'Dendritic': ['IRF7', 'IRF4', 'FCER1A', 'CD1C'],
    'Plasma': ['IGJ', 'MZB1'],
    'Cycling':['STMN1','TOP2A','MKI67']
}

# Plot a dotplot with the markers
sns.set_theme(style='white', font_scale=0.8)
markers = keep_specified_markers(refined_markers, adata.var_names.tolist())
sc.tl.dendrogram(adata, groupby='leiden', use_rep='X_pca_harmony', random_state=34524623)
fig, ax = plt.subplots(figsize=(8, 12))
sc.pl.dotplot(adata, markers, groupby='leiden', dendrogram=True, log=False, swap_axes=True, vmax=4, ax=ax, show=False)
plt.tight_layout()
plt.savefig('./plots/scs_dataset_dotplots/lyu_2024_celltype_marker_dotplot_harmony.png', dpi=120)
plt.savefig('./plots/scs_dataset_dotplots/lyu_2024_celltype_marker_dotplot_harmony.pdf')

# Figuring out missing cell type identities
cl = '20'
sc.tl.rank_genes_groups(adata, groupby='leiden', groups=[cl], layer='log1p',random_state=34524623)
deg_df = sc.get.rank_genes_groups_df(adata, group=cl, pval_cutoff=0.05, log2fc_min=1)
print(deg_df[:20])


# Annotate cell type clusters
celltype_annotation_dict = {
    'Epithelial': ['2','3','5','7','8','10','17','24','25','27','27','28','29','30','31'],
    'Endothelial': ['6','16','23'],
    'Fibroblast_muscle': ['11','14'],
    'Myeloid': ['4','21'],
    'T cell': ['0','1','9','13','15','19'],
    'B cell': ['12'],
    'Mast': ['18'],
    'Plasma':['22'],
    'Neuronal':['26'],
    'Cycling':['20']
    }


# Create a reverse mapping from number to cell type
number_to_celltype = {}
for celltype, numbers in celltype_annotation_dict.items():
    for number in numbers:
        number_to_celltype[number] = celltype

# Create the annotation column
adata.obs['celltype'] = adata.obs['leiden'].map(number_to_celltype)

# Plot an UMAP of the resulting annotation
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata, color='celltype', ax=ax, show=False, legend_loc="on data")
plt.tight_layout()
plt.savefig('./plots/scs_dataset_umaps/lyu_2024_celltype_umap_harmony.png', dpi=120)
plt.savefig('./plots/scs_dataset_umaps/lyu_2024_celltype_umap_harmony.pdf')

# Plot an UMAP of phenotype
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata, color='phenotype', ax=ax, show=False, legend_loc="on data")
plt.tight_layout()
plt.savefig('./plots/scs_dataset_umaps/lyu_2024_phenotype_umap_harmony.png', dpi=120)
plt.savefig('./plots/scs_dataset_umaps/lyu_2024_phenotype_umap_harmony.pdf')


# Plot an UMAP of sample
fig, ax = plt.subplots(figsize=(6, 4.5))
sc.pl.umap(adata, color='phenotype', ax=ax, show=False)#, legend_loc="on data")
plt.tight_layout()
plt.savefig('./plots/scs_dataset_umaps/lyu_2024_sample_umap_harmony.png', dpi=120)
plt.savefig('./plots/scs_dataset_umaps/lyu_2024_sample_umap_harmony.pdf')


# Save the annotated dataset
adata.write_h5ad('./single-cell/lyu_2024/adata_obj_harmony_annotated_20241217.h5ad')

# Subset epithelial and save
adata_epithelial = adata[adata.obs['celltype'] == 'Epithelial'].copy()
adata_epithelial.write_h5ad('./single-cell/lyu_2024/adata_obj_harmony_epithelial.h5ad')

# Subset Myeloid and save
adata_myeloid = adata[adata.obs['celltype'] == 'Myeloid'].copy()
adata_myeloid.write_h5ad('./single-cell/lyu_2024/adata_obj_harmony_myeloid.h5ad')

### FINISHED


####################################### Data preprocessing with own pipeline #######################################


##### QC & preprocessing #######
adata = sc.read_h5ad('./single-cell/lyu_2024/adata_obj.h5ad')

def qc_filters(adata, remove_doublets=True):
    # mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # Calculate the percentage
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=False)
    # Leave out cells with > 10% mitochondrial reads
    adata = adata[adata.obs.pct_counts_mt < 10, :] # 20
    # Filter out cells by using the same metrics as in Lyu et al. 2024
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_cells(adata, max_genes=2500)
    sc.pp.filter_genes(adata, min_cells= 3)
    sc.pp.filter_genes(adata, min_counts= 10)
    # Filter out cells with more than one HBB gene counts
    adata = adata[adata[:, 'HBB'].X <= 1] 
    if remove_doublets:
        sc.external.pp.scrublet(adata)
        adata = adata[adata.obs['predicted_doublet']==False]
    return adata

# Run filtering
adata = qc_filters(adata)
'''

Original unfiltered: 
236,856 cells × 33,538 genes

With the same filtering as in previous integration:

Automatically set threshold at doublet score = 0.84
Detected doublet rate = 0.0%
Estimated detectable doublet fraction = 0.9%
Overall doublet rate:
        Expected   = 5.0%
        Estimated  = 0.1%

Post filtering:
154,699 cells × 25,708 genes

Post filtering:
91,160 × 22,324

'''
#sc.pp.filter_genes(adata, min_counts= 10) # Not sure if ran properly as gave an error on 0 count genes?
# Run normalization
scib.preprocessing.normalize(adata,precluster=False, sparsify=False)
adata
# Save the normalized data object
adata.write_h5ad('./single-cell/lyu_2024/adata_obj_qc_normalized_orig.h5ad')

############################### Data analysis (post SCVI) ###############################

adata = sc.read_h5ad('./single-cell/lyu_2024/adata_obj_qc_normalized_orig_scvi_integrated.h5ad')

# Plot UMAP
sc.set_figure_params(figsize=(4, 4))
sc.pl.umap(adata, color=['VI_clusters', 'phenotype', 'sample'], show=False, save=True)

#### Create a dotplot for celltype annotation ####

# Define marker genes
def keep_specified_markers(marker_list, keep_vals):
    new_marker_list = {}
    for k, v in marker_list.items():
        new_marker_list[k] = [val for val in v if val in keep_vals]
    return new_marker_list

refined_markers = {
    'Epithelial': ['EPCAM', 'KRT18', 'KRT8', 'KLK3', 'AR', 'MSMB', 'KRT5', 'KRT15', 'TP63', 'KRT7', 'KRT19', 'KRT4'],
    'Endothelial': ['VWF', 'SELE', 'FLT1', 'ENG'],
    'Fibroblast': ['LUM', 'DCN', 'IGF1', 'APOD', 'STC1', 'FBLN1', 'COL1A2', 'C7', 'IGFBP5', 'ACTA2'],
    'SMC': ['RGS5', 'ACTA2', 'TAGLN', 'BGN', 'MYL9', 'MYLK', 'CALD1'],
    'Mast': ['KIT', 'TPSB2', 'TPSAB1', 'CPA3', 'VWA5A', 'IL1RL1', 'CTSG', 'SLC18A2', 'ACSL4', 'MS4A2', 'GATA2'],
    'T cell': ['CD3D', 'CD3E', 'CD3G', 'CD8A', 'CD8B', 'IL7R', 'NKG7', 'CD7', 'GNLY'],
    'B cell': ['CD79A', 'MS4A1', 'CD79B', 'IGHM', 'CD83'],
    'Myeloid': ['C1QA', 'C1QB', 'C1QC', 'CD68', 'LYZ', 'IL1A', 'IL1B', 'S100A9', 'S100A8', 'CXCL8', 'FCGR3A', 'CSF1R'],
    'Neuronal': ['PLP1', 'MPZ', 'MT1H'],
    'Dendritic': ['IRF7', 'IRF4', 'FCER1A', 'CD1C'],
    'Plasma': ['IGJ', 'MZB1']
}


# Plot a dotplot with the markers
sns.set_theme(style='white', font_scale=0.8)
markers = keep_specified_markers(refined_markers, adata.var_names.tolist())
sc.tl.dendrogram(adata, groupby='VI_clusters', use_rep='X_scVI', random_state=34524623)
fig, ax = plt.subplots(figsize=(8, 12))
sc.pl.dotplot(adata, markers, groupby='VI_clusters', dendrogram=True, log=False, swap_axes=True, vmax=4, ax=ax, show=False)
plt.tight_layout()
plt.savefig('./plots/scs_dataset_dotplots/lyu_2024_celltype_marker_dotplot_scvi.png', dpi=120)
plt.savefig('./plots/scs_dataset_dotplots/lyu_2024_celltype_marker_dotplot_scvi.pdf')

# Figuring out missing cell type identities
cl = '28'
sc.tl.rank_genes_groups(adata, groupby='VI_clusters', groups=[cl], random_state=34524623)
deg_df = sc.get.rank_genes_groups_df(adata, group=cl, pval_cutoff=0.05, log2fc_min=1)
print(deg_df[:20])

# Annotate cell type clusters
celltype_annotation_dict = {
    'Epithelial': ['1','3','12','15','18','23','24','26','27'],
    'Endothelial': ['5','17','22'],
    'Fibroblast_muscle': ['9','14'],
    'Myeloid': ['8','11','19'],
    'T cell': ['0','2','4','6','7','13','20'],
    'B cell': ['10'],
    'Mast': ['16'],
    'Plasma':['21'],
    'Neuronal':['25']
    }

# Create a reverse mapping from number to cell type
number_to_celltype = {}
for celltype, numbers in celltype_annotation_dict.items():
    for number in numbers:
        number_to_celltype[number] = celltype

# Create the annotation column
adata.obs['celltype'] = adata.obs['VI_clusters'].map(number_to_celltype)

# Plot an UMAP of the resulting annotation
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata, color='celltype', ax=ax, show=False, legend_loc="on data")
plt.tight_layout()
plt.savefig('./plots/scs_dataset_umaps/lyu_2024_celltype_umap_scvi.png', dpi=120)
plt.savefig('./plots/scs_dataset_umaps/lyu_2024_celltype_umap_scvi.pdf')

# Plot an UMAP of phenotype
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata, color='phenotype', ax=ax, show=False, legend_loc="on data")
plt.tight_layout()
plt.savefig('./plots/scs_dataset_umaps/lyu_2024_phenotype_umap_scvi.png', dpi=120)
plt.savefig('./plots/scs_dataset_umaps/lyu_2024_phenotype_umap_scvi.pdf')


# Plot an UMAP of sample
fig, ax = plt.subplots(figsize=(6, 4.5))
sc.pl.umap(adata, color='phenotype', ax=ax, show=False)#, legend_loc="on data")
plt.tight_layout()
plt.savefig('./plots/scs_dataset_umaps/lyu_2024_sample_umap_scvi.png', dpi=120)
plt.savefig('./plots/scs_dataset_umaps/lyu_2024_sample_umap_scvi.pdf')


# Save the annotated dataset
adata.write_h5ad('./single-cell/lyu_2024/adata_obj_scvi_annotated_20241212.h5ad')

# Subset epithelial and save
adata_epithelial = adata[adata.obs['celltype'] == 'Epithelial'].copy()
adata_epithelial.write_h5ad('./single-cell/lyu_2024/adata_obj_scvi_epithelial.h5ad')

# Subset Myeloid and save
adata_myeloid = adata[adata.obs['celltype'] == 'Myeloid'].copy()
adata_myeloid.write_h5ad('./single-cell/lyu_2024/adata_obj_scvi_myeloid.h5ad')


############################### Data analysis (without SCVI) ###############################
adata = sc.read_h5ad('./single-cell/lyu_2024/adata_obj_qc_normalized_orig.h5ad')
adata.uns['log1p']["base"] = None

# Set a global random seed
np.random.seed(435924236)

# Ensure var names are unique
adata.var_names_make_unique()

# Identify highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Perform PCA
sc.tl.pca(adata, random_state=435924236)

# Compute the neighborhood graph
sc.pp.neighbors(adata, random_state=435924236)

# Perform clustering
sc.tl.leiden(adata, key_added="leiden", resolution=1, random_state=435924236)

# Compute UMAP
sc.tl.umap(adata, random_state=435924236)

# Plot UMAP
sc.set_figure_params(figsize=(4, 4))
sc.pl.umap(adata, color=['leiden', 'phenotype', 'sample'], show=False, save=True)

#### Create a dotplot for celltype annotation ####

# Define marker genes
def keep_specified_markers(marker_list, keep_vals):
    new_marker_list = {}
    for k, v in marker_list.items():
        new_marker_list[k] = [val for val in v if val in keep_vals]
    return new_marker_list

refined_markers = {
    'Epithelial': ['EPCAM', 'KRT18', 'KRT8', 'KLK3', 'AR', 'MSMB', 'KRT5', 'KRT15', 'TP63', 'KRT7', 'KRT19', 'KRT4'],
    'Endothelial': ['VWF', 'SELE', 'FLT1', 'ENG'],
    'Fibroblast': ['LUM', 'DCN', 'IGF1', 'APOD', 'STC1', 'FBLN1', 'COL1A2', 'C7', 'IGFBP5', 'ACTA2'],
    'SMC': ['RGS5', 'ACTA2', 'TAGLN', 'BGN', 'MYL9', 'MYLK', 'CALD1'],
    'Mast': ['KIT', 'TPSB2', 'TPSAB1', 'CPA3', 'VWA5A', 'IL1RL1', 'CTSG', 'SLC18A2', 'ACSL4', 'MS4A2', 'GATA2'],
    'T cell': ['CD3D', 'CD3E', 'CD3G', 'CD8A', 'CD8B', 'IL7R', 'NKG7', 'CD7', 'GNLY'],
    'B cell': ['CD79A', 'MS4A1', 'CD79B', 'IGHM', 'CD83'],
    'Myeloid': ['C1QA', 'C1QB', 'C1QC', 'CD68', 'LYZ', 'IL1A', 'IL1B', 'S100A9', 'S100A8', 'CXCL8', 'FCGR3A', 'CSF1R'],
    'Neuronal': ['PLP1', 'MPZ', 'MT1H'],
    'Dendritic': ['IRF7', 'IRF4', 'FCER1A', 'CD1C'],
    'Plasma': ['IGJ', 'MZB1']
}


# Plot a dotplot with the markers
sns.set_theme(style='white', font_scale=0.8)
markers = keep_specified_markers(refined_markers, adata.var_names.tolist())
sc.tl.dendrogram(adata, groupby='leiden', use_rep='X_pca', random_state=34524623)
fig, ax = plt.subplots(figsize=(8, 12))
sc.pl.dotplot(adata, markers, groupby='leiden', dendrogram=True, log=False, swap_axes=True, vmax=4, ax=ax, show=False)
plt.tight_layout()
plt.savefig('./plots/scs_dataset_dotplots/lyu_2024_celltype_marker_dotplot.png', dpi=120)
plt.savefig('./plots/scs_dataset_dotplots/lyu_2024_celltype_marker_dotplot.pdf')

# Figuring out missing cell type identities
cl = '28'
sc.tl.rank_genes_groups(adata, groupby='leiden', groups=[cl], random_state=34524623)
deg_df = sc.get.rank_genes_groups_df(adata, group=cl, pval_cutoff=0.05, log2fc_min=1)
print(deg_df[:20])

# Annotate cell type clusters
celltype_annotation_dict = {
    'Epithelial': ['0','2','7','8','11','12','14','19','21','22','24','26','28','29','30','31','32','33','34'],
    'Endothelial': ['6','20','23'],
    'Fibroblast_muscle': ['10','18'],
    'Myeloid': ['9','13'],
    'T cell': ['1','3','4','5','16','17','27'],
    'B cell': ['15'],
    'Mast': ['25'],
    'drop': ['28'] # <-- HBB, HBA1, HBA2 red blood cells
    }

# Create a reverse mapping from number to cell type
number_to_celltype = {}
for celltype, numbers in celltype_annotation_dict.items():
    for number in numbers:
        number_to_celltype[number] = celltype

# Create the annotation column
adata.obs['celltype'] = adata.obs['leiden'].map(number_to_celltype)

# Drop the columns to drop
adata = adata[~(adata.obs['celltype'] == 'drop')]

# Plot an UMAP of the resulting annotation
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata, color='celltype', ax=ax, show=False, legend_loc="on data")
plt.tight_layout()
plt.savefig('./plots/scs_dataset_umaps/lyu_2024_celltype_umap.png', dpi=120)
plt.savefig('./plots/scs_dataset_umaps/lyu_2024_celltype_umap.pdf')

# Save the annotated dataset
adata.write_h5ad('./single-cell/lyu_2024/adata_obj_annotated_20241212.h5ad')

# Subset epithelial and save
adata_epithelial = adata[adata.obs['celltype'] == 'Epithelial'].copy()
adata_epithelial.write_h5ad('./single-cell/lyu_2024/adata_obj_epithelial.h5ad')

# Subset Myeloid and save
adata_myeloid = adata[adata.obs['celltype'] == 'Myeloid'].copy()
adata_myeloid.write_h5ad('./single-cell/lyu_2024/adata_obj_myeloid.h5ad')


