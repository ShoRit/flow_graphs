
import sys; sys.path.append('common');
from helper import *
from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertForNextSentencePrediction, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, IterableDataset
import fileinput

class BertDataset(Dataset):
	def __init__(self, dataset, num_class, tokenizer, params, data_idx=0, num_workers=None, lbl2id=None):
		self.dataset		= dataset
		self.num_class		= num_class
		self.p				= params
		self.tokenizer		= tokenizer
		self.data_idx 		= data_idx

		self.cls_tok 		= self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[CLS]'))
		self.sep_tok 		= self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[SEP]'))

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		ele = self.dataset[idx]
		return ele['tokens'], ele['labels'], ele['valid'], ele

	def pad_data(self, sents):
		max_len  = min(np.max([len(x[0]) for x in sents]), self.p.max_seq_len-2)

		tok_pad	 = np.zeros((len(sents), max_len+2), np.int32)		# +2 for ['CLS'] and ['SEP'] 
		tok_mask = np.zeros((len(sents), max_len+2), np.float32)	# +2 for ['CLS'] and ['SEP'] 
		tok_len	 = np.zeros((len(sents)), np.int32)
		val_pad	 = np.zeros((len(sents), max_len), np.float32)
		labels	 = np.zeros((len(sents), max_len, self.num_class), np.int32)

		for i, (toks, label, valid, _) in enumerate(sents):
			if len(toks) > max_len:
				toks    = toks  [: max_len]
				label   = label [: max_len]
				valid   = valid [: max_len]

			toks = self.cls_tok + toks + self.sep_tok

			tok_pad [i, :len(toks)]	 	= toks
			tok_mask[i, :len(toks)]		= 1.0
			tok_len [i]		 			= len(toks)
			val_pad [i, :len(valid)] 	= valid	

			for j, lbl in enumerate(label):
				labels[i, j, lbl] = 1.0

		return torch.LongTensor(tok_pad), torch.FloatTensor(tok_mask), torch.LongTensor(tok_len), torch.FloatTensor(val_pad), torch.LongTensor(labels)

	def collate_fn(self, all_data):
		all_data.sort(key = lambda x: -len(x[0]))

		batches = []
		num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

		for i in range(num_batches):
			start_idx  = i * self.p.batch_size
			data 	   = all_data[start_idx : start_idx + self.p.batch_size]

			tok_pad, tok_mask, tok_len, val_pad, labels = self.pad_data(data)
			if tok_pad is None: 
				print('MISS')
				continue

			batches.append ({
				'tok_pad'	: tok_pad,
				'tok_mask'	: tok_mask,
				'tok_len'	: tok_len,
				'val_pad'	: val_pad,
				'labels'	: labels,
				'_data_idx'   	: self.data_idx,
				'_rest'		: [x[-1] for x in data],
			})

		return batches

class BertDatasetFile(IterableDataset):
	def __init__(self, file_list, num_class, tokenizer, params, data_idx=0, num_workers = 0, lbl2id=None):
	
		self.p			= params
		self.num_class		= num_class
		self.num_workers	= num_workers
		self.file_list		= file_list
		self.lbl2id		= lbl2id
		self.data_idx 		= 0
		self.file_chunks	= partition(self.file_list, self.num_workers)

		self.tokenizer		= tokenizer
		self.cls_tok		= self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[CLS]'))
		self.sep_tok		= self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[SEP]'))

	def preprocess(self, line):
		try:
			doc  = json.loads(line)
			
		except Exception as e:
			print('\nException Cause: {}'.format(e.args[0]))
			return 0
		
		all_label = []
		mask = np.zeros(len(doc['labels']), np.int32)

		for i, label in enumerate(doc['labels']):
			if label != 'O':
				beg, lbls = label.split('-')
				lbls 	  = set(lbls.split('|'))
				mask[max(0, i-self.p.neg_sample): min(len(doc['labels'])-1, i+self.p.neg_sample)] = 1

				all_label.append([self.lbl2id[f'{beg}-{lbl}'] for lbl in lbls if ('all' in self.p.google_lbls) or (lbl in self.p.google_lbls)])
			else:
				all_label.append([self.lbl2id['O']])


		if all_label[:self.p.max_seq_len].count([0]) == len(all_label[:self.p.max_seq_len]): 
			return 0
		else: 					   
			return doc['tokens'], all_label, list(np.int32(doc['valid']) * mask), doc

	def __iter__(self):
		worker_info = torch.utils.data.get_worker_info()

		if worker_info is None: file_hand = fileinput.input(self.file_list)
		else: 			file_hand = fileinput.input(self.file_chunks[worker_info.id])

		return map(self.preprocess, file_hand)

	def pad_data(self, sents):
		max_len  = min(np.max([len(x[0]) for x in sents]), self.p.max_seq_len-2)

		tok_pad	 = np.zeros((len(sents), max_len+2), np.int32)		# +2 for ['CLS'] and ['SEP'] 
		tok_mask = np.zeros((len(sents), max_len+2), np.float32)	# +2 for ['CLS'] and ['SEP'] 
		tok_len	 = np.zeros((len(sents)), np.int32)
		val_pad	 = np.zeros((len(sents), max_len), np.float32)
		labels	 = np.zeros((len(sents), max_len, self.num_class), np.int32)

		for i, (toks, label, valid, _) in enumerate(sents):
			if len(toks) > max_len:
				toks    = toks  [: max_len]
				label   = label [: max_len]
				valid   = valid [: max_len]

			toks = self.cls_tok + toks + self.sep_tok

			tok_pad [i, :len(toks)]	 	= toks
			tok_mask[i, :len(toks)]		= 1.0
			tok_len [i]		 	= len(toks)
			val_pad [i, :len(valid)] 	= valid	

			for j, lbl in enumerate(label):
				labels[i, j, np.int32(lbl)] = 1.0

		return torch.LongTensor(tok_pad), torch.FloatTensor(tok_mask), torch.LongTensor(tok_len), torch.FloatTensor(val_pad), torch.LongTensor(labels)

	def collate_fn(self, all_data):
		all_data = [x for x in all_data if x != 0]
		all_data.sort(key = lambda x: -len(x[0]))

		batches = []
		num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

		for i in range(num_batches):
			start_idx  = i * self.p.batch_size
			data 	   = all_data[start_idx : start_idx + self.p.batch_size]

			tok_pad, tok_mask, tok_len, val_pad, labels = self.pad_data(data)

			batches.append ({
				'tok_pad'	: tok_pad,
				'tok_mask'	: tok_mask,
				'tok_len'	: tok_len,
				'val_pad'	: val_pad,
				'labels'	: labels,
				'_data_idx'   	: self.data_idx,
				'_rest'		: [x[-1] for x in data],
			})

		return batches
