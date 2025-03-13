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
""" 
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

 """
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


#### Calculate sample-specific percentages and save them #####

col_name = 'celltype'

obs_df = adata.obs.copy()
obs_df['count'] = 1 # Add a count term
cell_count = obs_df[['sample','count']].groupby('sample').sum()

# Need to have at least 40 cells per sample to be considered
#cell_count = cell_count[cell_count['count'] >= 20] # 50k epi --> 100, 10k myeloids --> 20
celltype_pct = pd.DataFrame(index = cell_count.index)

for cat in obs_df[col_name].cat.categories:
    df = obs_df[obs_df[col_name]==cat].copy()
    df = df[['sample','count']].groupby('sample').sum()
    df = df.loc[celltype_pct.index]
    celltype_pct[cat+' pct'] = (df['count'] / cell_count['count'])

# Convert to percentage
celltype_pct = celltype_pct * 100
if (celltype_pct.sum(axis=1).round(4)==100).all():
    print('Succesfully created percentage dataframe')

celltype_pct['phenotype'] = celltype_pct.index.map(obs_df[['sample','phenotype']].groupby('sample').first().to_dict()['phenotype'])
celltype_pct['phenotype'] = pd.Categorical(celltype_pct['phenotype'],categories=obs_df['phenotype'].cat.categories)

celltype_pct.to_csv('./lyu_2024_celltype_percentages.csv')

# Save the annotated dataset
adata.write_h5ad('./single-cell/lyu_2024/adata_obj_harmony_annotated_20241217.h5ad')

################################# Define and save subsets #################################
adata =  sc.read_h5ad('./single-cell/lyu_2024/adata_obj_harmony_annotated_20241217.h5ad')


#################### Epithelial cells ####################

# Subset epithelial and save
adata_epithelial = adata[adata.obs['celltype'] == 'Epithelial'].copy()

# Re-integrate based on just the epithelial cells
adata_epithelial.X = adata_epithelial.layers['log1p'].copy()

# Scale to uniform mean & variance
sc.pp.scale(adata_epithelial)

# Perform PCA on the dataframe
sc.pp.pca(adata_epithelial,n_comps=50)

# Correct components using Harmony
sc.external.pp.harmony_integrate(adata_epithelial,key='sample')

seed = 519552572

# Compute the neighborhood graph
sc.pp.neighbors(adata_epithelial, use_rep='X_pca_harmony', random_state=seed)
sc.tl.umap(adata_epithelial,random_state=seed)

sc.set_figure_params(figsize=(4, 4))
sc.pl.umap(adata_epithelial, color=['phenotype', 'sample'], show=False, save='_epithelial_clusters.png')


res = 1.0
sc.tl.leiden(adata_epithelial, key_added='leiden_epithelial', resolution=res, random_state=seed)
adata_epithelial.obs = adata_epithelial.obs.rename(columns={'leiden_epithelial':f'leiden_epithelial_res{res}'})

# Save
adata_epithelial.write_h5ad('./single-cell/lyu_2024/adata_obj_harmony_epithelial.h5ad')


#################### Myeloid cells ####################

# Subset Myeloid and save
adata_myeloid = adata[adata.obs['celltype'] == 'Myeloid'].copy()

# Re-integrate based on just the myeloid cells
adata_myeloid.X = adata_myeloid.layers['log1p'].copy()

# Scale to uniform mean & variance
sc.pp.scale(adata_myeloid)

# Perform PCA on the dataframe
sc.pp.pca(adata_myeloid,n_comps=50)

# Correct components using Harmony
sc.external.pp.harmony_integrate(adata_myeloid,key='sample')

seed = 519552572

# Compute the neighborhood graph
sc.pp.neighbors(adata_myeloid, use_rep='X_pca_harmony', random_state=seed)
sc.tl.umap(adata_myeloid,random_state=seed)

sc.set_figure_params(figsize=(4, 4))
sc.pl.umap(adata_myeloid, color=['phenotype', 'sample'], show=False, save='_myeloid_clusters.png')

res = 1.0
sc.tl.leiden(adata_myeloid, key_added=f'leiden_myeloid_res{res}', resolution=res, random_state=seed)

adata_myeloid.write_h5ad('./single-cell/lyu_2024/adata_obj_harmony_myeloid.h5ad')


#################### T cells ####################

# Subset T cells and save
adata_tcell = adata[adata.obs['celltype'] == 'T cell'].copy()

# Re-integrate based on just the T cells
adata_tcell.X = adata_tcell.layers['log1p'].copy()

# Scale to uniform mean & variance
sc.pp.scale(adata_tcell)

# Perform PCA on the dataframe
sc.pp.pca(adata_tcell,n_comps=50)

# Correct components using Harmony
sc.external.pp.harmony_integrate(adata_tcell,key='sample')

seed = 2534268

# Compute the neighborhood graph
sc.pp.neighbors(adata_tcell, use_rep='X_pca_harmony', random_state=seed)
sc.tl.umap(adata_tcell,random_state=seed)

sc.set_figure_params(figsize=(4, 4))
sc.pl.umap(adata_tcell, color=['phenotype', 'sample'], show=False, save='_T_cell_clusters.png')

res = 1.0
sc.tl.leiden(adata_tcell, key_added=f'leiden_t_cell_res{res}', resolution=res, random_state=seed)

adata_tcell.write_h5ad('./single-cell/lyu_2024/adata_obj_harmony_t_cell.h5ad')


### FINISHED
