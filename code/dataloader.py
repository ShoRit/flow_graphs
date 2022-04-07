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
		dep_data 		= ele['dep_data']  
		# node_vecs, edge_index, edge_type, n1_mask, n2_mask = ele['dep_data'].x, ele['dep_data'].edge_index, ele['dep_data'].edge_type, ele['dep_data'].n1_mask, ele['dep_data'].n2_mask      
		# node_vecs, edge_index, edge_type, n1_mask, n2_mask = torch.tensor(node_vecs), torch.tensor(edge_index), torch.tensor(edge_type), torch.tensor(n1_mask), torch.tensor(n2_mask)

		dep_data 		= Data(x=torch.tensor(dep_data.x), edge_index= torch.tensor(dep_data.edge_index), \
      						edge_type= torch.tensor(dep_data.edge_type), n1_mask=torch.tensor(dep_data.n1_mask), n2_mask=torch.tensor(dep_data.n2_mask))

		if self.domain 	== 'src': 	label 			= torch.tensor(self.rel2id[ele['label']])
		else:						label 			= None

		return (tokens, segments, marked_1, marked_2, desc_emb, label, dep_data)


class Bert_RGCN_RelDataset(Dataset):
	def __init__(self, dataset, rel2id, tokenizer, params, data_idx=0):
		self.dataset		= dataset
		self.rel2id			= rel2id
		self.p				= params
		self.tokenizer		= tokenizer
		self.data_idx 		= data_idx
		self.cls_tok 	    = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[CLS]'))
		self.sep_tok 	    = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[SEP]'))
		self.domain 		= self.p.domain

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		ele = self.dataset[idx]

		tokens 			= torch.tensor(self.cls_tok + ele['tokens'] + self.sep_tok)
		marked_1		= torch.tensor([0]+ele['arg1_ids']+[0])
		marked_2		= torch.tensor([0]+ele['arg2_ids']+[0])
		segments		= torch.tensor([0]*len(tokens))
		desc_emb		= ele['desc_emb']
		dep_data 		= ele['dep_data']  
		# node_vecs, edge_index, edge_type, n1_mask, n2_mask = ele['dep_data'].x, ele['dep_data'].edge_index, ele['dep_data'].edge_type, ele['dep_data'].n1_mask, ele['dep_data'].n2_mask      
		# node_vecs, edge_index, edge_type, n1_mask, n2_mask = torch.tensor(node_vecs), torch.tensor(edge_index), torch.tensor(edge_type), torch.tensor(n1_mask), torch.tensor(n2_mask)

		dep_data 		= Data(x=torch.tensor(dep_data.x), edge_index= torch.tensor(dep_data.edge_index), \
      						edge_type= torch.tensor(dep_data.edge_type), n1_mask=torch.tensor(dep_data.n1_mask), n2_mask=torch.tensor(dep_data.n2_mask))

		if self.domain 	== 'src': 	label 			= torch.tensor(self.rel2id[ele['label']])
		else:						label 			= None

		return (tokens, segments, marked_1, marked_2, desc_emb, label, dep_data)





	# def pad_data(self, sents):
	# 	max_len     = min(np.max([len(x[0]) for x in sents]), self.p.max_seq_len-2)

	# 	tok_pad	 	= np.zeros((len(sents), max_len+2), np.int32)		# +2 for ['CLS'] and ['SEP'] 
	# 	tok_mask 	= np.zeros((len(sents), max_len+2), np.float32)
	# 	tok_len	 	= np.zeros((len(sents)), np.int32)

	# 	arg1_ids 	= np.zeros((len(sents), max_len+2), np.int32)
	# 	arg2_ids 	= np.zeros((len(sents), max_len+2), np.int32)

	# 	labels	 	= np.zeros((len(sents), self.num_class), np.int32)
	# 	for i, (toks, arg1_id, arg2_id, label, feat, _) in enumerate(sents):
	# 		if len(toks) > max_len:
	# 			toks    = toks[: max_len]
	# 			arg1_id = arg1_id[: max_len]
	# 			arg2_id = arg2_id[: max_len]

	# 		toks = self.cls_tok + toks + self.sep_tok
	# 		# arg1_id = [0]+arg1_id +[0]
	# 		# arg2_id = [0]+arg2_id +[0]
	# 		tok_pad [i, :len(toks)]	 		= toks
	# 		tok_mask[i, :len(toks)]			= 1.0
	# 		tok_len [i]		 				= len(toks)
	# 		try:
	# 			arg1_ids[i, 1:len(arg1_id)+1]  	= arg1_id
	# 			arg2_ids[i, 1:len(arg2_id)+1]  	= arg2_id
	# 		except Exception as e:
	# 			import pdb; pdb.set_trace()
	# 		labels[i,] 						= label


	# 	return torch.LongTensor(tok_pad), torch.FloatTensor(tok_mask) , torch.LongTensor(tok_len), torch.FloatTensor(arg1_ids), torch.FloatTensor(arg2_ids), torch.FloatTensor(feat_arr), torch.LongTensor(labels)

	# def collate_fn(self, all_data):
	# 	all_data.sort(key = lambda x: -len(x[0]))

	# 	batches = []
	# 	num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

	# 	for i in range(num_batches):
	# 		start_idx  = i * self.p.batch_size
	# 		data 	   = all_data[start_idx : start_idx + self.p.batch_size]

	# 		tok_pad, tok_mask, tok_len, arg1_ids, arg2_ids, feat_arr, labels = self.pad_data(data)
	# 		if tok_pad is None: 
	# 			print('MISS')
	# 			continue

	# 		batches.append ({
	# 			'tok_pad'		: tok_pad,
	# 			'tok_mask'		: tok_mask,
	# 			'tok_len'		: tok_len,
	# 			'arg1_ids'		: arg1_ids,
	# 			'arg2_ids'		: arg2_ids,
	# 			'labels'		: labels,
	# 			'_data_idx'   	: self.data_idx,
	# 			'_rest'			: [x[-1] for x in data],
	# 		})

	# 	return batches

