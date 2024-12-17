import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
import pickle




def get_sample_ids_reorder(sample_types=['BPH','untreated','bicalutamide',
                    'goserelin','degarelix','degarelix_apalutamide','CRPC']):
    samples = []
    if 'BPH' in sample_types:
        samples = samples + ['BPH_651','BPH_665','BPH_688','BPH_652']

    if 'untreated' in sample_types:
        samples = samples + [
        # Untreated
        'PC-03-6712','PC_00_16338_II','PC_01_06342_VAS',
        'PC_01_14451_OIK','PC_02_05601_OIK','PC_02_10136_VAS',
        'PC_03_01669_TUTKV','PC_15420OIK','PC_4980','PC_7875OIK',
        'PC_05_08449_OIK_POST_0','PC_05_13057_OIK_ANT_0','PC_05_30627_VAS_ANT_0',
        'PC_06_02588_OIK_ANT_0','PC_06_04581_OIK_POST_0','PC_06_17800_VAS_POST_0',
        'PC_04_12646_VAS'
        ]

    if 'bicalutamide' in sample_types:
        samples = samples + [
        # Bicalutamide-treated
        'PC_05_16831_VAS_POST_1','PC_05_17668_OIK_ANT_1','PC_05_24402_OIK_ANT2_1',
        'PC_06_03093_OIK_ANT_1','PC_05_25866_OIK_POST_1','PC_06_11108_VAS_POST_1'
        ]

    if 'goserelin' in sample_types:
        samples = samples + [
        # Goserelin-treated
        'PC_05_27153_OIK_POST_2','PC_05_29927_VAS_POST_2','PC_06_00786_VAS_ANT_2',
        'PC_06_04077_OIK_ANT_2','PC_06_16086_VAS_POST_2'
        ]

    if 'degarelix' in sample_types:
        samples = samples + [
        # Degarelix-treated
        'P32030', 'P32033', 'P32037', 'P32043', 'P32055'
        ]

    if 'degarelix_apalutamide' in sample_types:
        samples = samples + [
        # Degarelix + apalutamide-treated
        'P32014', 'P32036', 'P32041', 'P32044', 'P32062', 'P32084'
        ]

    if 'CRPC' in sample_types:
        samples = samples + ['CRPC-278','CRPC-489','CRPC-697','CRPC_530','CRPC_531']

    return samples
""" 
def qc_and_normalize(adata):
    # requires scib-pipline-R4.0 conda environment !
    import scib
    # normalize and calculate leiden clustering
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.filter_cells(adata, min_counts=500) 
    scib.preprocessing.normalize(adata,precluster=False)
    return adata

def spatially_aware_clustering(adata,proximity_weight=0.3,res=1.0):
    import squidpy as sq
    # Define the joint adjacency weighting
    sc.pp.scale(adata)
    sc.pp.highly_variable_genes(adata)
    sc.pp.pca(adata, n_comps=15)
    sc.pp.neighbors(adata, random_state=36345)
    sq.gr.spatial_neighbors(adata, n_rings=2, coord_type="grid", n_neighs=6,transform='cosine')
    joint_adj = adata.obsp['spatial_connectivities']*proximity_weight + adata.obsp['connectivities']
    sc.tl.leiden(adata,adjacency=joint_adj,key_added='joint_leiden_clusters',resolution=res,random_state=36345)
    return adata

 """

def save_to_pickle(obj,filename):
    import pickle
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_from_pickle(filename):
    import pickle
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def get_treatment_info():
    samples = {}
    samples['BPH_651'] = 'bph'
    samples['BPH_665'] = 'bph'
    samples['BPH_688'] = 'bph'
    samples['BPH_652'] = 'bph'
    samples['PC-03-6712'] = 'untreated'
    samples['PC_00_16338_II'] = 'untreated'
    samples['PC_01_06342_VAS'] = 'untreated'
    samples['PC_01_14451_OIK'] = 'untreated'
    samples['PC_02_05601_OIK'] = 'untreated'
    samples['PC_02_10136_VAS'] = 'untreated'
    samples['PC_03_01669_TUTKV'] = 'untreated'
    samples['PC_15420OIK'] = 'untreated'
    samples['PC_4980'] = 'untreated'
    samples['PC_7875OIK'] = 'untreated'
    samples['PC_05_08449_OIK_POST_0'] = 'untreated'
    samples['PC_05_13057_OIK_ANT_0'] = 'untreated'
    samples['PC_05_16831_VAS_POST_1'] = 'bicalutamide'
    samples['PC_05_27153_OIK_POST_2'] = 'goserelin'
    samples['PC_05_17668_OIK_ANT_1'] = 'bicalutamide'
    samples['PC_05_24402_OIK_ANT2_1'] = 'bicalutamide'
    samples['PC_05_29927_VAS_POST_2'] = 'goserelin'
    samples['PC_05_30627_VAS_ANT_0'] = 'untreated'
    samples['PC_05_25866_OIK_POST_1'] = 'bicalutamide'
    samples['PC_06_00786_VAS_ANT_2'] = 'goserelin'
    samples['PC_06_02588_OIK_ANT_0'] = 'untreated'
    samples['PC_06_03093_OIK_ANT_1'] = 'bicalutamide'
    samples['PC_06_04077_OIK_ANT_2'] = 'goserelin'
    samples['PC_06_04581_OIK_POST_0'] = 'untreated'
    samples['PC_06_11108_VAS_POST_1'] = 'bicalutamide'
    samples['PC_06_16086_VAS_POST_2'] = 'goserelin'
    samples['PC_06_17800_VAS_POST_0'] = 'untreated'
    samples['PC_04_12646_VAS'] = 'untreated'
    samples['P32014'] = 'degarelix_apalutamide'
    samples['P32030'] = 'degarelix'
    samples['P32033'] = 'degarelix'
    samples['P32036'] = 'degarelix_apalutamide'
    samples['P32037'] = 'degarelix'
    samples['P32041'] = 'degarelix_apalutamide'
    samples['P32043'] = 'degarelix'
    samples['P32044'] = 'degarelix_apalutamide'
    samples['P32055'] = 'degarelix'
    samples['P32062'] = 'degarelix_apalutamide'
    samples['P32084'] = 'degarelix_apalutamide'
    samples['CRPC-278'] = 'crpc'
    samples['CRPC-489'] = 'crpc'
    samples['CRPC-697'] = 'crpc'
    samples['CRPC_531'] = 'crpc'
    samples['CRPC_530'] = 'crpc'

    return samples

def get_sample_crop_coords():
    coords = {'BPH_651':(2500,19000,3000,19000),
     'BPH_665':(4000,20400,3100,19300),
     'BPH_688':(2900,19400,2900,19000),
     'BPH_652':(2000,18500,2800,19000),
     'PC-03-6712':(2200,18500,2900,19000),
     'PC_00_16338_II':(0,16400,1600,17800),
     'PC_01_06342_VAS':(1500,17900,2300,18500),
     'PC_01_14451_OIK':(2000,18300,6200,22300),
     'PC_02_05601_OIK':(2200,18700,4000,20200),
     'PC_02_10136_VAS':(4600,21000,3000,19200),
     'PC_03_01669_TUTKV':(500,16900,4100,20400),
     'PC_15420OIK':(2200,18600,2800,19000),
     'PC_4980':(900,16800,800,17000),
     'PC_7875OIK':(2500,19000,2700,18700),
     'PC_05_08449_OIK_POST_0':(1000,17300,1100,17400),
     'PC_05_13057_OIK_ANT_0':(800,17200,2100,18200),
     'PC_05_16831_VAS_POST_1':(800,17200,1700,17800),
     'PC_05_27153_OIK_POST_2':(1900,18300,6600,22800),
     'PC_05_17668_OIK_ANT_1':(2900,19300,3100,19300),
     'PC_05_24402_OIK_ANT2_1':(3000,19300,2800,18900),
     'PC_05_29927_VAS_POST_2':(2100,18500,4000,20100),
     'PC_05_30627_VAS_ANT_0':(2900,19000,13900,29900),
     'PC_05_25866_OIK_POST_1':(6600,22900,15300,31500),
     'PC_06_00786_VAS_ANT_2':(1900,18500,14900,31000),
     'PC_06_02588_OIK_ANT_0':(3000,19300,15300,31500),
     'PC_06_03093_OIK_ANT_1':(800,17200,5100,21200),
     'PC_06_04077_OIK_ANT_2':(2500,18900,8600,24700),
     'PC_06_04581_OIK_POST_0':(2900,19300,8400,24400),
     'PC_06_11108_VAS_POST_1':(2300,18700,11200,27400),
     'PC_06_16086_VAS_POST_2':(2800,19100,4100,20100),
     'PC_06_17800_VAS_POST_0':(1600,18000,3200,19400),
     'PC_04_12646_VAS':(1900,18300,5700,21800),
     'P32014':(0,0,0,0),
     'P32030':(0,0,0,0),
     'P32033':(0,0,0,0),
     'P32036':(0,0,0,0),
     'P32037':(0,0,0,0),
     'P32041':(0,0,0,0),
     'P32043':(0,0,0,0),
     'P32044':(0,0,0,0),
     'P32055':(0,0,0,0),
     'P32062':(0,0,0,0),
     'P32084':(0,0,0,0),
     'CRPC-278':(2400,18800,2800,18900),
     'CRPC-489':(2700,18100,3400,19500),
     'CRPC-697':(2100,18500,3100,19200),
     'CRPC_531':(2200,18600,7500,23600),
     'CRPC_530':(2000,18400,3300,19400),
     'MET_A3':(2000,18400,3300,19400),
     'MET_GP12':(2000,18400,3300,19400),
     'MET_A14':(2000,18400,3300,19400),
     'MET_A16':(2000,18400,3300,19400),
        }
    return(coords)


def get_sample_id_mask():
    df = pd.read_csv('./patient_id_mask.csv',sep=';')
    mask_dict = dict(zip(df['PAD'],df['Pseudo_ID']))
    return(mask_dict)

def get_include_exclude_info():
    df = pd.read_csv('./concat_exclude_information_Tampere_ARNEO.csv',index_col=0)
    return(df)