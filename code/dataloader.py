import imp
from helper import *
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_geometric.data import Data


class ZSBertRelDataset(Dataset):
	def __init__(self, dataset, rel2id, tokenizer, params, data_idx=0, domain='src'):
		self.dataset		= dataset
		self.rel2id			= rel2id
		self.p				= params
		self.tokenizer		= tokenizer
		self.data_idx 		= data_idx
		self.cls_tok 	    = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[CLS]'))
		self.sep_tok 	    = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[SEP]'))
		self.domain 		= domain

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		ele = self.dataset[idx]

		tokens 			= torch.tensor(self.cls_tok + ele['tokens'] + self.sep_tok)
		marked_1		= torch.tensor([0]+ele['arg1_ids']+[0])
		marked_2		= torch.tensor([0]+ele['arg2_ids']+[0])
		segments		= torch.tensor([0]*len(tokens))
		desc_emb		= ele['desc_emb']

		if self.domain 	== 'src': 	label 			= torch.tensor(self.rel2id[ele['label']])
		else:						label 			= None

		return (tokens, segments, marked_1, marked_2, desc_emb, label)


class ZSBert_RGCN_RelDataset(Dataset):
	def __init__(self, dataset, rel2id, tokenizer, params, data_idx=0, domain='src'):
		self.dataset		= dataset
		self.rel2id			= rel2id
		self.p				= params
		self.tokenizer		= tokenizer
		self.data_idx 		= data_idx
		self.domain 		= domain

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		ele = self.dataset[idx]

		tokens 			= torch.tensor(ele['tokens'] +  [0]*(self.p.max_seq_len-len(ele['tokens'])))
		marked_1		= torch.tensor(ele['arg1_ids']+ [0]*(self.p.max_seq_len - len(ele['tokens'])))
		marked_2		= torch.tensor(ele['arg2_ids']+ [0]*(self.p.max_seq_len - len(ele['tokens'])))
		segments		= torch.tensor([0]*len(tokens))
		desc_emb		= ele['desc_emb']
		dep_data 		= ele['dep_data']  
		# node_vecs, edge_index, edge_type, n1_mask, n2_mask = ele['dep_data'].x, ele['dep_data'].edge_index, ele['dep_data'].edge_type, ele['dep_data'].n1_mask, ele['dep_data'].n2_mask      
		# node_vecs, edge_index, edge_type, n1_mask, n2_mask = torch.tensor(node_vecs), torch.tensor(edge_index), torch.tensor(edge_type), torch.tensor(n1_mask), torch.tensor(n2_mask)

		dep_data 		= Data(x=torch.tensor(dep_data.x), edge_index= torch.tensor(dep_data.edge_index), \
      						edge_type= torch.tensor(dep_data.edge_type), n1_mask=torch.tensor(dep_data.n1_mask), n2_mask=torch.tensor(dep_data.n2_mask))

		# dep_data 		= Data(x=dep_data.x.clone().detach().requires_grad_(True), edge_index= dep_data.edge_index.clone().detach(), \
      	# 					edge_type=dep_data.edge_type.clone().detach(), n1_mask=dep_data.n1_mask.clone().detach(),\
        #         			n2_mask=dep_data.n2_mask.clone().detach())


		# import pdb; pdb.set_trace()

		if self.domain 	== 'src': 	label 			= torch.tensor(self.rel2id[ele['label']])
		else:						label 			= None

		return (tokens, segments, marked_1, marked_2, desc_emb, label, dep_data)

