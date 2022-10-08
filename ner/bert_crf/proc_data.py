import sys
from helper import *
from intervals import *
import spacy

def proc_ann(ann_fname, lbl_remap = {}):
	ann_data = []

	for line in open(ann_fname):
		index, info, word	= line.split('\t')
		label, start, end 	= [x for x in info.split() if ';' not in x]
		start, end 			= int(start), int(end)
		# label, rest 		= info.split(' ')[0], ' '.join(info.split(' ')[1:])
		# start_end_off 		= [[int(x.split(' ')[0]), int(x.split(' ')[1])] for x in rest.split(';')]

		if start == end or word.strip() == '':
			continue

		ann_data.append({
			'_id'		: index,
			'start'		: start,
			'end'		: end,
			'word'		: word.strip(),
			'label'		: lbl_remap.get(label, label)
		})

	return ann_data

def get_dataset_raw(data_dir, dataset, tokenizer, lbl2id, mode=None, bert_long=-1, text_tokenizer='scispacy'):
	data   = ddict(list)
	labels = {}
	if text_tokenizer == 'scispacy': nlp = spacy.load("en_core_sci_md")

	count = 0 
	for split in ['test','valid','train']:
		for root, dirs, files in os.walk(f'{data_dir}/{dataset}/{split}'):
			for file in files:
				if not file.endswith('.ann'): continue
				ann_fname = os.path.join(root, file)
				txt_fname = ann_fname.replace('.ann', '.txt')
				doc_id 	  = file.replace('.ann', '')
				text = open(txt_fname).read()
				anns = proc_ann(ann_fname)
				ann_map = IntervalMapping()
				lbl_cnt = ddict(int)
				for ann in anns:
					ann_map[ann['start']: ann['end']] = (ann['label'], (ann['start'], ann['end']))			
				if   text_tokenizer == 'scispacy': sents = [x for x in nlp(text).sents]
				# elif text_tokenizer == 'chemext':  sents = Paragraph(doc['text']).sentences

				for sent in sents:
					tokens, tok_range, org_toks, labels, valid, org_labels = [], [], [], [], [], []

					if   text_tokenizer == 'scispacy': 		sent_toks = sent
					# elif text_tokenizer == 'chemext':  sent_toks = sent.tokens

					for tok in sent_toks:
						if   text_tokenizer == 'scispacy': 	start, end = tok.idx, tok.idx + len(tok)

						assert text[start: end] == tok.text

						if ann_map.contains(start, end): 
							_lbl, off = ann_map.get_label(start, end)
							lbl = ('B-' if off not in lbl_cnt else 'I-') + _lbl
							lbl_cnt[off] += 1
						else:
							lbl = 'O'

						label = lbl2id.get(lbl, 0)

						bert_toks = tokenizer.encode_plus(tok.text,  add_special_tokens=False)['input_ids']

						if bert_long != -1 and len(bert_toks) >= bert_long:
							bert_toks = bert_toks[:bert_long] + tokenizer.encode_plus('[LONG_TOKEN]',  add_special_tokens=False)['input_ids']
							# bert_toks = tokenizer.encode_plus('[LONG_TOKEN]',  add_special_tokens=False)['input_ids']					

						if len(bert_toks) != 0:
							tokens    += bert_toks
							tok_range.append([start, end])
							org_toks.append(tok.text)

							labels     += [label] * len(bert_toks)
							org_labels += [lbl] * len(bert_toks)
							valid 	   += [1] + [0] * (len(bert_toks) - 1)

					if len(tokens) ==0: continue

					data[split].append({
						'tokens'	: tokens,
						'labels'	: labels,
						'org_labels'	: org_labels,
						'valid'		: valid,
						'file'		: file,
						'tok_range'	: tok_range,
						'org_toks'	: org_toks,
					})
					# import pdb; pdb.set_trace()
				count+=1
				if count % 100 == 0: print('Completed {}, {}'.format(count, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))
	return data



def get_test_dataset(pred_dir, tokenizer,  lbl2id, mode=None, bert_long=-1, text_tokenizer='scispacy'):
	data   				= ddict(list)
	labels 				= {}
	if text_tokenizer == 'scispacy': nlp = spacy.load("en_core_sci_md")
	count 				= 0 

	for root, dirs, files in os.walk(f'{pred_dir}'):
		for file in files:
			if not file.endswith('.txt'): continue
			txt_fname = os.path.join(root, file)
			doc_id 	  = file.replace('.txt', '')
			text = open(txt_fname).read()
			
			if   text_tokenizer == 'scispacy': sents = [x for x in nlp(text).sents]
			# elif text_tokenizer == 'chemext':  sents = Paragraph(doc['text']).sentences

			for sent in sents:
				tokens, tok_range, org_toks, labels, valid, org_labels = [], [], [], [], [], []

				if   text_tokenizer == 'scispacy': sent_toks = sent
				# elif text_tokenizer == 'chemext':  sent_toks = sent.tokens

				for tok in sent_toks:
					if   text_tokenizer == 'scispacy': start, end = tok.idx, tok.idx + len(tok)
					# elif text_tokenizer == 'chemext':  start, end = tok.start, tok.end
					assert text[start: end] == tok.text

					lbl = 'O'

					label = lbl2id.get(lbl, 0)

					bert_toks = tokenizer.encode_plus(tok.text,  add_special_tokens=False)['input_ids']

					if bert_long != -1 and len(bert_toks) >= bert_long:
						bert_toks = bert_toks[:bert_long] + tokenizer.encode_plus('[LONG_TOKEN]',  add_special_tokens=False)['input_ids']
						# bert_toks = tokenizer.encode_plus('[LONG_TOKEN]',  add_special_tokens=False)['input_ids']					

					if len(bert_toks) != 0:
						tokens    += bert_toks
						tok_range.append([start, end])
						org_toks.append(tok.text)

						labels     += [label] * len(bert_toks)
						org_labels += [lbl] * len(bert_toks)
						valid 	   += [1] + [0] * (len(bert_toks) - 1)

				data['test'].append({
					'tokens'	: tokens,
					'labels'	: labels,
					'org_labels'	: org_labels,
					'valid'		: valid,
					'file'		: file,
					'tok_range'	: tok_range,
					'org_toks'	: org_toks,
				})
				# import pdb; pdb.set_trace()
			count+=1
			if count % 100 == 0: print('Completed {}, {}'.format(count, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))
	
	return data

	



if __name__== "__main__":

	bert_model 		= 	'bert-base-cased'
	bert_tokenizer  = 	BertTokenizerFast.from_pretrained(bert_model)
	lbl2id 			= 	json.load(open('config/risec_lbl2id.json'))
 
	get_dataset_raw('/projects/flow_graphs/data/','risec', bert_tokenizer, lbl2id)