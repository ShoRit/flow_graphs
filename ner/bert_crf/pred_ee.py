from models_ee import *
from helper import *
# from proc_data import *
from proc_data import *
from dataloader_ee import *
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import average_precision_score, roc_auc_score, auc, f1_score, confusion_matrix
# import evaluation
# from ner_evaluation.ner_eval import *

class PredEE(object):

	def load_data(self):
		# Load label mapping
		self.lbl2id		= json.load(open(f'{self.p.config_dir}/{self.p.data}_lbl2id.json'))
		self.id2lbl		= {v: k for k, v in self.lbl2id.items()}
		self.num_class	= len(self.lbl2id)

		self.gaz2id		= {'B-GAZ':1, 'I-GAZ':2, 'O':0}
		self.id2gaz		= {v: k for k, v in self.gaz2id.items()}

		
		self.tokenizer  = BertTokenizerFast.from_pretrained(self.p.bert_dir)

		if self.p.long != -1:
			self.tokenizer.add_special_tokens({'additional_special_tokens': ["[LONG_TOKEN]"]})

		if self.p.pred_only is not None:
			self.data = get_test_dataset(self.p.pred_dir, self.tokenizer, self.lbl2id, mode=self.p.mode, bert_long=self.p.long, text_tokenizer=self.p.text_tok)
		
		else:
			# Load data
			fname = f'{self.p.cache_dir}/{self.p.data}|{self.p.bert_model.replace("/", "_")}|{self.p.text_tok}|{self.p.long}.pkl'

			if not check_file(fname) or self.p.mode is not None:
				self.data = get_dataset_raw(self.p.data_dir ,self.p.data, self.tokenizer, self.lbl2id, mode=self.p.mode, bert_long=self.p.long, text_tokenizer=self.p.text_tok)
				
				if not self.p.sample:
					dump_pickle(self.data, fname)
			else:
				self.data = load_pickle(fname)
			

			self.data['train'], self.data['valid'], self.data['test'] = self.data['train'], self.data['valid'], self.data['test']
  	
  
		self.total_data = len(self.data['train'])
		self.logger.info('\nDataset size -- Train: {}, Valid: {}, Test:{}'.format(len(self.data['train']), len(self.data['valid']), len(self.data['test'])))

		def get_data_loader(split, dataClass, shuffle=True):
			dataset	= dataClass(self.data[split], self.num_class, self.tokenizer, self.p, num_workers=self.p.num_workers)
			return DataLoader(
				dataset,
				batch_size      = self.p.batch_size * self.p.batch_factor,
				num_workers     = self.p.num_workers,
				collate_fn      = dataset.collate_fn
			)

		self.data_iter = {
			'train'	: get_data_loader('train', BertDataset),
			'valid'	: get_data_loader('valid', BertDataset, shuffle=False),
			'test'	: get_data_loader('test',  BertDataset, shuffle=False),
		}

	def add_model(self):
		if 	self.p.model == 'bert': 	model = BertPlain(self.p, self.num_class)
		else:	raise NotImplementedError

		if self.p.long:
			model.bert.resize_token_embeddings(len(self.tokenizer))

		model = model.to(self.device)

		if len(self.gpu_list) > 1:
			print ('Using multiple GPUs ', self.p.gpu)
			model = nn.DataParallel(model, device_ids = list(range(len(self.p.gpu.split(',')))))
			torch.backends.cudnn.benchmark = True

		return model

	def add_optimizer(self, model, train_dataset_length):
		if self.p.opt == 'adam':
			warmup_proportion 	= 0.1
			n_train_steps		= int(train_dataset_length / self.p.batch_size ) * self.p.max_epochs
			num_warmup_steps	= int(float(warmup_proportion) * float(n_train_steps))

			param_optimizer		= list(model.named_parameters())

			if self.p.model == 'bert_comb' and not self.p.no_fix_bert:
				param_optimizer = [x for x in param_optimizer if 'bert' not in x[0]]

			param_optimizer		= [n for n in param_optimizer if 'pooler' not in n[0]]
			no_decay		= ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

			optimizer_grouped_parameters = [
				{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
				{'params': [p for n, p in param_optimizer if     any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
			]

			optimizer = AdamW(optimizer_grouped_parameters, lr=self.p.lr)
			scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=n_train_steps)
			return optimizer, scheduler
		else: 
			return torch.optim.SGD(model.parameters(),  lr=self.p.lr, weight_decay=self.p.l2), None

	def __init__(self, params):
		self.p = params

		self.save_dir = f'{self.p.model_dir}/{self.p.log_db}'
		self.dump_dir = f'{self.p.pred_dir}/{self.p.name}/{self.p.data}'
		if not os.path.exists(self.p.log_dir): os.system(f'mkdir -p {self.p.log_dir}')		# Create log directory if doesn't exist
		if not os.path.exists(self.save_dir):  os.system(f'mkdir -p {self.save_dir}')		# Create model directory if doesn't exist
		if not os.path.exists(self.dump_dir):  os.system(f'mkdir -p {self.dump_dir}')

		self.p.cmd = os.popen(f"cat /proc/{os.getpid()}/cmdline | xargs -0 echo").read().strip()

		# Get Logger
		# self.mongo_log	= ResultsMongo(self.p)
		self.logger	= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
		self.logger.info(vars(self.p)); pprint(vars(self.p))
		self.logger.info(self.p.cmd)

		self.gpu_list = self.p.gpu.split(',')
		
		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		# self.device = torch.device('cpu')

		self.load_data()
		self.model			= self.add_model()
		self.optimizer,self.scheduler	= self.add_optimizer(self.model, self.total_data)

	def load_model(self, load_path):
		if self.p.copy:	state = torch.load('{}/{}'.format(load_path, self.p.copy),map_location=torch.device('cpu'))
		else: 		state = torch.load('{}/{}'.format(load_path, self.p.name),map_location=torch.device('cpu'))

		self.best_val		= 0.0
		self.best_test		= 0.0
		self.best_epoch		= 0

		if len(self.gpu_list) > 1:
			
			state_dict 	= state['state_dict']
			new_state_dict  = OrderedDict()

			for k, v in state_dict.items():
				if 'module' not in k:
					k = 'module.' + k
				else:
					k = k.replace('features.module.', 'module.features.')
				new_state_dict[k] = v

			if self.p.restore_google:
				own_state = self.model.state_dict()
				del new_state_dict['classifier.weight']
				del new_state_dict['classifier.bias']

				for name, param in new_state_dict.items():
					if name not in own_state:
						 continue
					if isinstance(param, Parameter):
						param = param.data
					own_state[name].copy_(param)
			else:
				self.model.load_state_dict(new_state_dict)
		else:
			state_dict 	= state['state_dict']
			new_state_dict  = OrderedDict()

			for k, v in state_dict.items():
				if 'module' in k:
					k = k.replace('module.', '')

				new_state_dict[k] = v

			if self.p.restore_google:
				own_state = self.model.state_dict()
				del new_state_dict['classifier.weight']
				del new_state_dict['classifier.bias']

				for name, param in new_state_dict.items():
					if name not in own_state:
						 continue
					if isinstance(param, Parameter):
						param = param.data
					own_state[name].copy_(param)
			else:
				self.model.load_state_dict(new_state_dict)

		if self.p.restore_opt:
			self.optimizer.load_state_dict(state['optimizer'])
			self.best_test	= state['best_test']
			self.best_val	= state['best_val']
			self.best_epoch	= state['best_epoch']

	def save_model(self, save_path):
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_test'	: self.best_test,
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
		}
		torch.save(state, '{}/{}'.format(save_path, self.p.name))

	def analyse_misclassification(self, true_dir, pred_dir):

		true_files = sorted(glob(true_dir+'/*.ann'))
		pred_files = sorted(glob(pred_dir+'/*.ann'))

		print(len(true_files), len(pred_files))

		exact_match     = ddict(lambda : ddict(list))
		incorrect_span  = ddict(lambda : ddict(list))
		incorrect_type  = ddict(lambda : ddict(list))
		

		for tfname, pfname in zip(true_files, pred_files):
			
			tf = open(tfname)
			pf = open(pfname)

			tdict = ddict(list); pdict = ddict(list); ttype_dict = ddict(list); ptype_dict = ddict(list)

			for line in tf:
				tid, tinfo, ttoken = line.strip().split('\t')
				ttype, spos, epos  = tinfo.strip().split()
				tdict[ttype].append((int(spos), int(epos), ttoken))
				ttype_dict[f'{ttoken}-{spos}-{epos}']= ttype

			for line in pf:
				tid, tinfo, ttoken = line.strip().split('\t')
				ttype, spos, epos  = tinfo.strip().split()
				pdict[ttype].append((int(spos), int(epos), ttoken))
				ptype_dict[f'{ttoken}-{spos}-{epos}']= ttype

			for key in tdict.keys():
				tvals = sorted(tdict[key])
				try:
					pvals = sorted(pdict[key])
					common = list(set(tvals)& set(pvals))
					exact_match[key][tfname].extend(common)

					missed_tps = set(tvals)-set(common)
					incorrect_tps = set(pvals)-set(common)

					if len(missed_tps) >0:
						incorrect_span[key][tfname].append(set(tvals)-set(common))
					else:
						incorrect_span[key][tfname].append(None)

					if len(incorrect_tps)>0:
						incorrect_span[key][tfname].append(set(pvals)-set(common))
					else:
						incorrect_span[key][tfname].append(None)

				except Exception as e:
					continue

			for token_ in ttype_dict:
				if token_ in ptype_dict:
					if ptype_dict[token_] != ttype_dict[token_]:
						incorrect_type[ttype_dict[token_]][ptype_dict[token_]].append((token_, tfname))


		# pdb.set_trace()
		

	def ee_evaluate(self, ann_dict, true_dir, pred_dir, split):
		if len(ann_dict) == 0: return;

		for root, dirs, files in os.walk(true_dir):
			for file in files:
				if not file.endswith('.ann'):
					continue
				ann_fname = os.path.join(root, file)
				txt_fname = ann_fname.replace('.ann', '.txt')

				shutil.copy(txt_fname, pred_dir)

		f1_score = 0.0
		def get_list(ttype, end_flag= True):
			# if ttype in ['REAGENT_CATALYST', 'REACTION_PRODUCT', 'STARTING_MATERIAL', 'OTHER_COMPOUND', 'SOLVENT']:
			# 	return [',', ' ', '.', ':']

			if end_flag == True:
				return [',', ')', ' ', '.', ']', '}', ':']
			else:
				return [',', '(', ' ', '.', '{', '[', ':']

		for file in ann_dict:
			
			span_fp = open(f'{pred_dir}/{file.replace(".ann", ".txt")}').read()
			fp      = open(f'{pred_dir}/{file}', 'w')

			for i, (label, start, end, wrd, emb) in enumerate(ann_dict[file]):
				if not self.p.no_post:
					if wrd[-1] in get_list(label, True):
						wrd  = str(wrd[:-1])
						end -= 1

					if len(wrd) == 0: continue;

					if wrd[0] in get_list(label, False):
						wrd    = str(wrd[1:])
						start += 1

					if start >= end: continue

					org_span = str(span_fp[start: end])
					if wrd.strip() == org_span:
						wrd = org_span
					else:
						copy_org_span = org_span.replace(' ','')
						copy_span     = wrd.replace(' ','')
						if copy_span == copy_org_span:
							wrd = org_span

				fp.write(f'T{i}\t{label} {start} {end}\t{wrd}\n')

			fp.close()
			
		dump_pickle(ann_dict, f'{pred_dir}_emb.pkl')


	def pred_evaluate(self, ann_dict, true_dir, pred_dir, split):
		if len(ann_dict) == 0: return;

		for file in glob(f'{true_dir}/*.txt'):
			if not file.endswith('.txt'):
				continue
			# txt_fname 		= file.split('/')[-1]
			shutil.copy(file, pred_dir)


		f1_score = 0.0
		def get_list(ttype, end_flag= True):
			# if ttype in ['REAGENT_CATALYST', 'REACTION_PRODUCT', 'STARTING_MATERIAL', 'OTHER_COMPOUND', 'SOLVENT']:
			# 	return [',', ' ', '.', ':']

			if end_flag == True:
				return [',', ')', ' ', '.', ']', '}', ':']
			else:
				return [',', '(', ' ', '.', '{', '[', ':']

		for file in ann_dict:
			try:
				if file.endswith('.txt') == False:  continue
				span_fp = open(f'{pred_dir}/{file}').read()
				fp      = open(f'{pred_dir}/{file.replace(".txt",".ann")}', 'w')

				for i, (label, start, end, wrd, emb) in enumerate(ann_dict[file]):
					if not self.p.no_post:
						if wrd[-1] in get_list(label, True):
							wrd  = str(wrd[:-1])
							end -= 1

						if len(wrd) == 0: continue;

						if wrd[0] in get_list(label, False):
							wrd    = str(wrd[1:])
							start += 1

						if start >= end: continue

						org_span = str(span_fp[start: end])
						if wrd.strip() == org_span:
							wrd = org_span
						else:
							copy_org_span = org_span.replace(' ','')
							copy_span     = wrd.replace(' ','')
							if copy_span == copy_org_span:
								wrd = org_span

					fp.write(f'T{i}\t{label} {start} {end}\t{wrd}\n')

				fp.close()
			except Exception as e:
				import pdb; pdb.set_trace()

		dump_pickle(ann_dict, f'{pred_dir}_emb.pkl')




	def get_acc(self, split, logits, labels, valid, all_rest=None, all_toks = None):
		mask = np.concatenate([np.reshape(x, -1) for x in valid], axis=0)
		if np.sum(mask) == 0: return None

		all_logits = np.concatenate([np.reshape(x, [-1, self.num_class]) for x in logits], axis=0)[mask == 1].argmax(1)
		all_labels = np.concatenate([np.reshape(x, [-1, self.num_class]) for x in labels], axis=0)[mask == 1].argmax(1)

		if self.p.metric == 'macro':
			result = np.round(f1_score(all_labels, all_logits, average='macro'), 3)

		elif self.p.metric == 'brat':
			if all_rest is None: 
				result = np.round(f1_score(all_labels, all_logits, average='macro'), 3)
			else:
				os.system(f'rm -rf {self.dump_dir}/*')
				ann_dict = {}

				for i in range(len(all_rest)):
					for j in range(logits[i].shape[0]):

						logit, val, rest, embs = logits[i][j], valid[i][j], all_rest[i][j], all_toks[i][j]
						
						emb_arr   = []
						curr_emb  = np.zeros(768)
						start_flag = True

						for idx in range(0, len(val)):
							if val[idx] == 1 and start_flag:
								curr_emb = embs[idx]
								start_flag = False
							elif val[idx] == 1 and not start_flag:
								emb_arr.append(curr_emb)
								curr_emb = embs[idx]
							if val[idx] == 0:
								curr_emb+= embs[idx]

						if len(embs) >0:
							emb_arr.append(curr_emb)

						pred      = logit.argmax(1)[val == 1.0]

						assert len(pred) == len(emb_arr)

						tok_range = rest['tok_range']
						toks 	  = rest['org_toks']

						if rest['file'] not in ann_dict:
							ann_dict[rest['file']] = []

						start, label, prev_end, wrd, emb, num_words = None, None, None, '', np.zeros(768), 1
						for k in range(len(pred)):
							lbl = self.id2lbl[pred[k]]

							if lbl == 'O': 
								if start is None: continue
								else:
									ann_dict[rest['file']].append((label, start, prev_end, wrd,emb/num_words))
									start, label, prev_end, wrd, emb, num_words = None, None, None, '', np.zeros(768), 1

							elif lbl.startswith('B-'):
								if start is not None:
									ann_dict[rest['file']].append((label, start, prev_end, wrd, emb))

								start, prev_end	= tok_range[k]
								label		= lbl.split('-')[1]
								wrd			= toks[k]
								emb     	= emb_arr[k]
								num_words   = 1

							elif lbl.startswith('I-'):
								if label == lbl.split('-')[1]:
									prev_end 	= tok_range[k][1]
									wrd	 		=  wrd + ' ' + toks[k]
									emb 		= emb + emb_arr[k]
									num_words   +=1
								else:
									start, prev_end	= tok_range[k]
									label		= lbl.split('-')[1]
									wrd			= toks[k]
									emb 		= emb_arr[k]
									num_words   = 1

						if start is not None:
							start, label, prev_end, wrd, emb, num_words = None, None, None, '', np.zeros(768), 1

				if self.p.pred_only is None:
					self.ee_evaluate(ann_dict, f'{self.p.data_dir}/{self.p.data}/{split}', self.dump_dir, split)
				else:
					self.pred_evaluate(ann_dict, f'{self.p.pred_dir}', self.dump_dir, split)
					exit(0)

				# results = os.popen(f'java -cp /home/u783945/ee-dow/brat_eval/nicta_biomed-brateval-41ce89f42038/target/brateval-0.3.0-SNAPSHOT.jar au.com.nicta.csp.brateval.CompareEntities \
				# 			{self.dump_dir} ./data/{self.p.data}/{split} {self.p.exact}').read().strip().split('\n')

				results = os.popen(f'java -cp ./brat_eval/target/BRATEval-0.1.0-SNAPSHOT.jar au.com.nicta.csp.brateval.CompareEntities \
							{self.dump_dir} {self.p.data_dir}/{self.p.data}/{split} {self.p.exact}').read().strip().split('\n')
				


				print(results[-1])
				result  = float(results[-1].split('|')[-1])

				print(result)
				self.analyse_misclassification(f'{self.p.data_dir}/{self.p.data}/{split}', self.dump_dir)

				return result
		else:
			raise NotImplementedError

		return result

	def execute(self, batch):
		batch		= to_gpu(batch, self.device)
		loss, logits, tok_embs 	= self.model(batch)

		if len(self.gpu_list) > 1: loss = loss.mean()

		return loss, logits, tok_embs

	def predict(self, epoch, split, return_extra=False):
		self.model.eval()

		all_eval_loss, all_logits, all_labels, all_valid, all_rest, all_toks, cnt = [], [], [], [], [], [],  0

		bad_bat, good_bat = 0,0
		with torch.no_grad():

			for batches in self.data_iter[split]:
				for k, batch in enumerate(batches):
					eval_loss, logits, tok_embs = self.execute(batch)
					if eval_loss== None:
						bad_bat +=1
						continue
					good_bat+=1
					all_eval_loss.append(eval_loss.item())
					all_logits.append(logits.detach().cpu().numpy())
					all_labels.append(batch['labels'].cpu().numpy())
					all_valid.append(batch['val_pad'].cpu().numpy())
					all_toks.append(tok_embs.cpu().numpy())
					all_rest.append(batch['_rest'])
					cnt += batch['tok_len'].shape[0]
		
		print(bad_bat, good_bat)

		eval_res = self.get_acc(split, all_logits, all_labels, all_valid, all_rest=all_rest, all_toks= all_toks)
		self.logger.info('Final Performance {} --> Loss: {:.3}, Acc: {:.3}'.format(split, np.mean(all_eval_loss), eval_res))

		if return_extra: return np.mean(all_eval_loss), eval_res, all_logits, all_labels, all_valid, all_rest, all_toks
		else: 		 return np.mean(all_eval_loss), eval_res

	def check_and_save(self, epoch):
		valid_loss, valid_acc = self.predict(epoch, 'valid')

		if valid_acc > self.best_val:
			self.best_val		= valid_acc
			_, self.best_test	= self.predict(epoch, 'test')
			self.best_epoch		= epoch
			if self.p.nosave is False:
				self.save_model(self.save_dir)
			return True
	
		return False


	def run_epoch(self, epoch, shuffle=True):
		self.model.train()

		all_train_loss, all_score, cnt, tot_batch   = 0, 0, 0, 0
		all_logits, all_labels, all_valid, all_rest = [], [], [], []
		save_cnt = 1

		for batches in self.data_iter['train']:
			for k, batch in enumerate(batches):
				self.optimizer.zero_grad()

				train_loss, logits, tok_embs = self.execute(batch)

				if (k+1) % self.p.log_freq == 0:
					eval_res = np.round(all_score / tot_batch, 3)

					self.logger.info('[E: {}] | {:.3}% | {} | L: {:.3}, T: {}, B-V:{}, B-T:{}'.format(epoch, \
						100*cnt/self.total_data, self.p.name, all_train_loss / tot_batch, eval_res, self.best_val, self.best_test))

				if (cnt/self.total_data) > (self.p.save_freq * save_cnt):
					self.check_and_save(epoch)
					save_cnt += 1

				try:
					all_train_loss += train_loss.item()
				except Exception as e:
					import pdb; pdb.set_trace()

				scr = self.get_acc('train', [logits.detach().cpu().numpy()], [batch['labels'].cpu().numpy()], [batch['val_pad'].cpu().numpy()])
				if scr is not None: all_score += scr

				train_loss.backward()
				self.optimizer.step()
				self.scheduler.step()

				cnt		+= batch['tok_len'].shape[0]
				tot_batch	+= 1

				
		return np.round(all_train_loss / tot_batch, 3), np.round(all_score / tot_batch, 3)


	def display_pred(self, logits, rest):
		colors	= ['head', 'info', 'warn', 'succ', 'fail']

		for i, b_rest in enumerate(rest): 
			b_pred  = logits[i].argmax(2)
			for j, sent in enumerate(b_rest):
				valid = np.int32(sent['valid'][: self.p.max_seq_len]) == 1
				pred  = b_pred[j][valid]
				for k in range(len(pred)):
					tok_pred    = pred[k]
					if tok_pred == self.lbl2id['O']:        print(sent['org_toks'][k], end=' ')
					else:                                   print(color('{}|{}'.format(sent['org_toks'][k], self.id2lbl[pred[k]]), colors[tok_pred % len(colors)]), end=' ')

				print()


	def fit(self):
		self.best_val, self.best_test, self.best_epoch = 0.0, 0.0, 0

		if self.p.restore:
			self.load_model(self.save_dir)

			if self.p.mode == 'man':
				_, _, logits, _, rest	= self.predict(0, 'train', return_extra=True)
				self.display_pred(logits, rest)

			if self.p.pred_only is not None:
				test_loss, test_acc, logits, labels, mask, rest, tok_embs = self.predict(0, self.p.pred_only, return_extra=True)
				import pdb; pdb.set_trace()

			if self.p.eval_only is not None:
				test_loss, test_acc, logits, labels, mask, rest, tok_embs = self.predict(0, self.p.eval_only, return_extra=True)
				self.logger.info('Performance on Test: {}'.format(test_acc))

				all_mask   = np.concatenate([np.reshape(x, -1) for x in mask], axis=0)
				all_logits = np.concatenate([np.reshape(x, [-1, self.num_class]) for x in logits], axis=0)[all_mask == 1].argmax(1)
				all_labels = np.concatenate([np.reshape(x, [-1, self.num_class]) for x in labels], axis=0)[all_mask == 1].argmax(1)

				# np.round(f1_score(all_labels, all_logits, average='macro'), 3) np.unique(all_logits)
				self.logger.info('\n' + classification_report(all_labels, all_logits, target_names=[f'{self.id2lbl[x]}_{x}' for x in range(self.num_class)], labels=np.unique(all_labels), digits=3))
				conf_max = confusion_matrix(all_labels, all_logits, labels=np.unique(all_labels))
				# self.display_pred(logits, rest)
				exit(0)

			if self.p.dump_only is not None:
				all_logits, all_labels, all_rest = [], [], []

				for split in self.p.dump_only.split(','):

					loss, acc, logits, labels, mask, rest, tok_embs = self.predict(0, split, return_extra=True)

					# self.display_pred(logits, rest)
					
					self.logger.info('Score {}: Loss: {}, Acc:{}'.format(split, loss, acc))

					all_logits	+= logits
					all_labels	+= labels
					all_rest	+= rest

					self.logger.info(classification_report(np.concatenate(labels, 0), np.concatenate(logits, 0) > 0.5, target_names=[self.id2lbl[x] for x in range(self.num_class)], digits=3))

				dump_dir = './predictions/{}'.format(self.p.data); make_dir(dump_dir)
				dump_pickle({
					'logits': np.concatenate(all_logits, axis=0),
					'labels': np.concatenate(all_labels, axis=0),
					'others': all_rest
				}, '{}/{}'.format(dump_dir, self.p.name))

				exit(0)

		kill_cnt = 0
		for epoch in range(self.p.max_epochs):
			train_loss, train_acc	= self.run_epoch(epoch)

			if self.check_and_save(epoch): 
				kill_cnt = 0
			else:
				kill_cnt += 1
				if kill_cnt > self.p.kill_cnt:
					self.logger.info('Early Stopping!')
					break

			# self.mongo_log.add_results(self.best_val, self.best_test, self.best_epoch, train_loss)

		self.logger.info('Best Performance: {}'.format(self.best_test)) 

if __name__== "__main__":

	parser = argparse.ArgumentParser(description='BERT-based-CRF')

	parser.add_argument('--gpu',      	default='0',                				help='GPU to use')
	parser.add_argument("--model", 		default='bert', 	type=str, 			help='Model for training and inference')
	parser.add_argument("--embed", 	 	default="bert_nospk_mean_sec_last", 	type=str, 	help="bert, biobert")

	# Bert
	parser.add_argument('--max_seq_len', 	default=512, 		type=int, 			help='Max allowed length of utt')
	parser.add_argument('--bert_model', 	default='bert-base-cased', 		type=str, 	help='Used for initializing BERT in the architecture. Can be any model from HuggingFace-Transformers')
	parser.add_argument('--bert_dir', 		default='bert_dir', 		    type=str, 	help='Used for initializing BERT in the architecture. Can be any model from HuggingFace-Transformers')
	parser.add_argument('--data', 	 		default='risec', 			type=str, 	help='Which data')
	parser.add_argument('--data_dir', 		default='/projects/flow_graphs/data/ner', 		    type=str, 	help='Used for initializing BERT in the architecture. Can be any model from HuggingFace-Transformers')

	# Attn
	parser.add_argument('--kill_cnt',    	dest='kill_cnt',	default=5,    	type=int,       help='Max epochs')
	parser.add_argument('--epoch',    	dest='max_epochs',	default=300,    type=int,       help='Max epochs')
	parser.add_argument('--batch',    	dest='batch_size',	default=8,      type=int,      	help='Batch size')
	parser.add_argument('--batch_factor',   dest='batch_factor',	default=50,     type=int,      	help='Number of batches to generate at one time')
	parser.add_argument('--num_workers',	type=int,		default=0,                   	help='Number of cores used for preprocessing data')
	parser.add_argument('--opt',      	default='adam',             				help='Optimizer to use for training')
	parser.add_argument('--lr', 	 	default=1e-3, 		type=float, 			help='The initial learning rate for Adam.')
	parser.add_argument('--l2', 	 	default=0.0, 		type=float, 			help='The initial learning rate for Adam.')
	parser.add_argument('--drop', 	 	default=0.1, 		type=float, 			help='The initial learning rate for Adam.')

	parser.add_argument('--comb_opn',    	default='concat',   	     				help='Experiment name')

	parser.add_argument('--log_db',    	default='main',   	     				help='Experiment name')
	parser.add_argument('--seed',     	default=1234,   	type=int,       		help='Seed for randomization')
	parser.add_argument('--log_freq',    	default=10,   		type=int,     			help='Display performance after these number of batches')
	parser.add_argument('--save_freq',    	default=1.0,   		type=float,     		help='Display performance after these number of batches')
	parser.add_argument('--name',     	default='test',             				help='Name of the run')
	parser.add_argument('--restore',  				action='store_true',        	help='Restore from the previous best saved model')
	parser.add_argument('--retrain',  				action='store_true',        	help='Restore from the previous best saved model')
	parser.add_argument('--restore_opt',  				action='store_true',        	help='Restore Optimizer from the previous best saved model')
	parser.add_argument('--restore_google',  			action='store_true',        	help='Restore Optimizer from the previous best saved model')
	parser.add_argument('--no_fix_bert',  				action='store_true',        	help='Restore Optimizer from the previous best saved model')
	parser.add_argument('--sample',  				action='store_true',        	help='Restore Optimizer from the previous best saved model')

	parser.add_argument('--mode',  		default=None,        					help='')
	parser.add_argument('--alpha',    	default=1.0,   		type=float,     		help='Display performance after these number of batches')

	parser.add_argument('--lstm',  					action='store_true',        	help='Restore Optimizer from the previous best saved model')
	parser.add_argument('--crf',  					action='store_true',        	help='Restore Optimizer from the previous best saved model')
	parser.add_argument('--gaz',  					action='store_true',        	help='Restore Optimizer from the previous best saved model')
	parser.add_argument('--hier',  					action='store_true',        	help='Restore Optimizer from the previous best saved model')
	parser.add_argument('--nosave',  				action='store_true',        	help='Restore Optimizer from the previous best saved model')
	parser.add_argument('--exact',  				action='store_true',        	help='Restore Optimizer from the previous best saved model')
	parser.add_argument('--no_post',  				action='store_true',        	help='Restore Optimizer from the previous best saved model')

	parser.add_argument('--dump_only',  			default=None,        					help='Dump logits of test dataset')
	parser.add_argument('--eval_only',  			default=None,        					help='Dump logits of test dataset')
	parser.add_argument('--dump_toks',  			default=None,        					help='Dump token embeddings of test dataset')
	parser.add_argument('--pred_only', 				default=None,							help='Dump predictions of the file')

	parser.add_argument('--type_cnt',  				default='final',	type=str,        	help='Number of Coarse grained types to use')
	parser.add_argument('--metric',  				default='brat',	type=str,        		help='Metric to use')
	parser.add_argument('--text_tok',  				default='scispacy',	type=str,        	help='Metric to use')

	# LSTM
	parser.add_argument('--rnn_layers',     default=2,   		type=int,       		help='')
	parser.add_argument('--rnn_dim',     	default=256,   		type=int,       		help='')
	parser.add_argument('--long',     		default=-1,   		type=int,       		help='')

	parser.add_argument('--cache_dir',   	default='./cache',        				help='Cache directory')
	parser.add_argument('--config_dir',   	default='./config',        				help='Config directory')
	parser.add_argument('--model_dir',   	default='./models',        				help='Model directory')
	parser.add_argument('--log_dir',   		default='./log',   	   				help='Log directory')
	parser.add_argument('--pred_dir',   	default='./res_files', 				help='Directory containing the predicted and true labels')
	

	parser.add_argument('--copy',  			default=None,	type=str,        		help='Metric to use')
	args = parser.parse_args()
	set_gpu(args.gpu)

	if not args.restore or args.copy:
		args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S') + '_' + str(uuid.uuid4())[:4]

	# Set seed
	np.random.seed(args.seed)
	random.seed(args.seed)
	torch.manual_seed(args.seed)

	# Create Model
	model = PredEE(args)
	model.fit()

	if args.retrain:
		del model
		torch.cuda.empty_cache()
		args.restore		= True
		args.lr			= 1e-5
		args.restore_opt	= True
		args.copy		= None

		model = PredEE(args)
		model.fit()

	print('Model Trained Successfully!!')
