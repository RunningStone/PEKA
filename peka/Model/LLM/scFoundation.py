from peka import logger
from peka.Model.LLM.base import scLLM_QC_preprocess, filter_zero_row

########################################################################################
# utils
########################################################################################
import scFoundation

import random
import numpy as np
import pandas as pd

import torch
from scipy.sparse import issparse
from anndata import AnnData

from peka.External_models.scFoundation.model.pretrainmodels.select_model import select_model
def convertconfig(ckpt):
    newconfig = {}
    newconfig['config']={}
    model_type = ckpt['config']['model']
    
    for key, val in ckpt['config']['model_config'][model_type].items():
        newconfig['config'][key]=val
        
    for key, val in ckpt['config']['dataset_config']['rnaseq'].items():
        newconfig['config'][key]=val
        
    if model_type == 'performergau_resolution':
        model_type = 'performer_gau'
    
    import collections
    d = collections.OrderedDict()
    for key, val in ckpt['state_dict'].items():
        d[str(key).split('model.')[1]]=val
        
    newconfig['config']['model_type']=model_type
    newconfig['model_state_dict']=d
    newconfig['config']['pos_embed']=False
    newconfig['config']['device']='cuda'
    return newconfig

def load_model_frommmf(best_ckpt_path, key='gene'):
    model_data = torch.load(best_ckpt_path,map_location='cpu')
    model_data = model_data[key]
    model_data = convertconfig(model_data)
    if not model_data.__contains__('config'):
        print('***** No config *****')
        config={}
        config['model_type']='flash_all'
    else:
        config=model_data['config']
        print(config)
    if not config.__contains__('qv_dim'):
        if config['model'] != 'mae_autobin':
            if config.__contains__('dim_head'):
                config['qv_dim']=config['dim_head']
            else:
                print('***** No qv_dim ***** set 64')
                config['qv_dim']= 64
    if not config.__contains__('ppi_edge'):
        config['ppi_edge']=None
    model = select_model(config)
    model_state_dict = model_data['model_state_dict']    
    model.load_state_dict(model_state_dict)
    return model.cuda(),config

def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein 
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene 
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                              columns=to_fill_columns, 
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns,var

def gatherDatanopad(data, labels, pad_token_id):
    max_num = labels.sum(1)
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = torch.tensor(-float('Inf'), device=labels.device)

    tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
    labels += tmp_data


    fake_label_gene_idx = labels.topk(max_num).indices

    new_data = torch.gather(data, 1, fake_label_gene_idx)

    padding_labels = (new_data == pad_token_id)

    return new_data, padding_labels

def gatherData(data, labels, pad_token_id):
    value_nums = labels.sum(1)
    max_num = max(value_nums, 2)


    fake_data = torch.full((data.shape[0], max_num), pad_token_id,
                           device=data.device)
    data = torch.hstack([data, fake_data])

    fake_label = torch.full((labels.shape[0], max_num), 1,
                            device=labels.device)
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = torch.tensor(-float('Inf'), device=labels.device)

    tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
    labels += tmp_data

    labels = torch.hstack([labels, fake_label])

    fake_label_gene_idx = labels.topk(max_num).indices

    new_data = torch.gather(data, 1, fake_label_gene_idx)

    padding_labels = (new_data == pad_token_id)

    return new_data, padding_labels


def convert_to_scFoundation_vocab_length(gexpr_feature:AnnData, gene_list):

    idx = gexpr_feature.obs_names.tolist()
    col = gexpr_feature.var_names.tolist()

    if issparse(gexpr_feature.X):
        gexpr_feature = gexpr_feature.X.toarray()
    else:
        gexpr_feature = gexpr_feature.X
    gexpr_feature = pd.DataFrame(gexpr_feature,index=idx,columns=col)

    if gexpr_feature.shape[1]<19264 or gexpr_feature.shape[1]>=19264:
        print('covert gene feature into 19264')
        gexpr_feature, to_fill_columns,var = main_gene_selection(gexpr_feature,gene_list)
        assert gexpr_feature.shape[1]>=19264

    return gexpr_feature


def preprocess_data_for_scFoundation(i, gexpr_feature, preprocess_type, tgthighres):
    #Single cell mode 
    # pre-Normalization
    if preprocess_type == 'F':
        tmpdata = (np.log1p(gexpr_feature.iloc[i,:]/(gexpr_feature.iloc[i,:].sum())*1e4)).tolist()
    elif preprocess_type == 'T':
        tmpdata = (gexpr_feature.iloc[i,:]).tolist()
    elif preprocess_type == 'A':
        tmpdata = (gexpr_feature.iloc[i,:-1]).tolist()
    else:
        raise ValueError('pre_normalized must be T,F or A')

    if preprocess_type == 'A':
        totalcount = gexpr_feature.iloc[i,-1]
    else:
        totalcount = gexpr_feature.iloc[i,:].sum()

    # select resolution
    if tgthighres[0] == 'f':
        pretrain_gene_x = torch.tensor(tmpdata+[np.log10(totalcount*float(tgthighres[1:])),np.log10(totalcount)]).unsqueeze(0).cuda()
    elif tgthighres[0] == 'a':
        pretrain_gene_x = torch.tensor(tmpdata+[np.log10(totalcount)+float(tgthighres[1:]),np.log10(totalcount)]).unsqueeze(0).cuda()
    elif tgthighres[0] == 't':
        pretrain_gene_x = torch.tensor(tmpdata+[float(tgthighres[1:]),np.log10(totalcount)]).unsqueeze(0).cuda()
    else:
        raise ValueError('tgthighres must be start with f, a or t')

    return pretrain_gene_x


########################################################################################
# scFoundation
########################################################################################
class scFoundation_embedder(scLLM_QC_preprocess):
    def __init__(self,
                 data_root,dataset_name,scLLM_embedder_name:str,
                 ckpt_name:str='default_model',
                 model_mode = "gene",):
        super().__init__(data_root,dataset_name,scLLM_embedder_name,ckpt_name)

        self.set_random(seed_nb=2024)
        # set gene list
        ckpt_path = self.pretrained_ckpt_dir + "/default_model.ckpt"
        vocab_path = self.pretrained_ckpt_dir + "/OS_scRNA_gene_index.19264.tsv"
        self.get_gene_vocab(vocab_path)
        self.pretrainmodel,self.pretrainconfig = load_model_frommmf(ckpt_path,model_mode)
        self.pretrainmodel.eval()

        logger.info(f" 🤖 embedding with scFoundation model {ckpt_name}..")

    def set_random(self,seed_nb):
        #Set random seed
        random.seed(seed_nb)
        np.random.seed(seed_nb)  # numpy random generator

        torch.manual_seed(seed_nb)
        torch.cuda.manual_seed_all(seed_nb)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_gene_vocab(self,gene_vocab_file):
        gene_list_df = pd.read_csv(gene_vocab_file, header=0, delimiter='\t')
        self.gene_list = list(gene_list_df['gene_name'])

    def model_related_qc_step(self, adata):
        filter_flag,_ = filter_zero_row(adata)
        adata.obs['filter_flag'] = filter_flag
        return adata, filter_flag
    
    def model_related_embed_infer_step(self, 
                                       adata, 
                                        preprocess_type = "F", # "F" not normalised, "T" normalised, "A" log normalised
                                        tgthighres = "f1", # T=number (starting with 't'), fold change of high resolution which means T/S=number (starting with 'f'), 
                                                            # or addition of high resolution which means T=S+number (starting with 'a'). 
                                        pool_type = "all", # "max" max pooling, "all" all pooling
                                        ):
        
        geneexpemb=[]
        gexpr_feature = convert_to_scFoundation_vocab_length(adata, self.gene_list)
        #Inference
        for i in range(gexpr_feature.shape[0]):
            with torch.no_grad():
                pretrain_gene_x = preprocess_data_for_scFoundation(i, gexpr_feature, preprocess_type, tgthighres)
                data_gene_ids = torch.arange(19266, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)

                value_labels = pretrain_gene_x > 0
                value_nums = value_labels.sum(1)
                max_num = max(value_nums)
                if max_num >2:
                    x, x_padding = gatherData(pretrain_gene_x, value_labels, self.pretrainconfig['pad_token_id'])
                    #print(f" data shape in x {x.shape}and x_padding {x_padding.shape}")
                    #Cell embedding
                    position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.pretrainconfig['pad_token_id'])
                    x = self.pretrainmodel.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
                    position_emb = self.pretrainmodel.pos_emb(position_gene_ids)
                    #print(f" before pos+x x shape {x.shape} and pos {position_emb.shape}")
                    x += position_emb
                    geneemb = self.pretrainmodel.encoder(x,x_padding)
                    #print(f" get geneemb shape {geneemb.shape}")
                    geneemb1 = geneemb[:,-1,:]
                    geneemb2 = geneemb[:,-2,:]
                    geneemb3, _ = torch.max(geneemb[:,:-2,:], dim=1)
                    geneemb4 = torch.mean(geneemb[:,:-2,:], dim=1)
                    if pool_type=='all':
                        geneembmerge = torch.concat([geneemb1,geneemb2,geneemb3,geneemb4],axis=1)
                    elif pool_type=='max':
                        geneembmerge, _ = torch.max(geneemb, dim=1)
                    else:
                        raise ValueError('pool_type must be all or max')
                    geneexpemb.append(geneembmerge.detach().cpu().numpy())
                else:
                    if pool_type=='all':
                        geneembmerge = torch.concat([geneemb1,geneemb1,geneemb1,geneemb1],axis=1)
                    elif pool_type=='max':
                        geneembmerge = geneemb1
                    else:
                        raise ValueError('pool_type must be all or max')
                    geneexpemb.append(geneembmerge.detach().cpu().numpy())
        geneexpemb = np.squeeze(np.array(geneexpemb))
        return geneexpemb