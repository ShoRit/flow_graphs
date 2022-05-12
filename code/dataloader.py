import imp
from collections import defaultdict

from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel

from helper import *



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

class ZSBert_RGCN_AMR_Dataset(Dataset):
	def __init__(self, annotated_instances, tokenizer):
		self.annotated_instances = annotated_instances
		self.tokenizer = tokenizer

	def __getitem__(self, idx):
		instance = self.annotated_instances[idx]
		
		node_to_idx = {"head_node": 0}
		tokenized_input_ids = [101]
		edges = []
		edge_types = []
		node_to_token = defaultdict(list)

		for sentence in instance:
			aligned_graph = sentence["graph"]
			tokens = sentence["tokens"]

			if aligned_graph is None:
				for (s, r, t) in aligned_graph.triples:
					if s not in node_to_idx:
						node_to_idx[s] = len(node_to_idx)
					if t not in node_to_idx:
						node_to_idx[t] = len(node_to_idx)
				token_to_node = defaultdict(list)

				alignments = penman.surface.alignments(aligned_graph)
				for triple in aligned_graph.triples:
					s,r,t = triple 
					if triple in alignments:
						for token_idx in alignments[triple].indices:
							token_to_node[token_idx].append(node_to_idx[s])
							token_to_node[token_idx].append(node_to_idx[t])
					edges.append((node_to_idx[s], node_to_idx[t]))
					edge_types.append(edge_to_type[r])
				# add an edge linking to the top node across sentences
				edges.append((0, node_to_idx[aligned_graph.top]))

			for i, token in enumerate(tokens):
				tokenized = tokenizer(token, add_special_tokens=False)["input_ids"]
				current_idx = len(tokenized_input_ids)
				tokenized_input_ids.extend(tokenized)
				if i in token_to_node:
					node_indices= token_to_node[i]
					for node_idx in node_indices:
						node_to_token[node_idx].extend(range(current_idx, current_idx + len(tokenized)))

		tokenized_input_ids.append(102)
		node_to_token = dict(node_to_token)