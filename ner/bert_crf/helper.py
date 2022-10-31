import os, sys, pdb, numpy as np, random, argparse, codecs, pickle, time, json, csv, itertools, bz2, re, requests, uuid, dill, ast
import logging, logging.config, itertools, pathlib, socket, warnings, pandas, shutil

from tqdm import tqdm
from glob import glob
from pprint import pprint
from pymongo import MongoClient
from itertools import chain
from scipy.stats import describe
from collections import OrderedDict
from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.exceptions import UndefinedMetricWarning
from collections import defaultdict as ddict, Counter
from sklearn.metrics import f1_score, classification_report, roc_auc_score, precision_score, precision_recall_fscore_support

# Pytorch related imports
import torch, torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter as Param
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertTokenizer, BertModel, BertConfig, BertPreTrainedModel, AutoTokenizer, AutoModel, BertTokenizerFast
from transformers.optimization import AdamW

import cProfile, pstats, io

def profile(fnc):
	
	"""A decorator that uses cProfile to profile a function"""
	
	def inner(*args, **kwargs):
		
		pr = cProfile.Profile()
		pr.enable()
		retval = fnc(*args, **kwargs)
		pr.disable()
		s = io.StringIO()
		sortby = 'cumulative'
		ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
		ps.print_stats()
		print(s.getvalue())
		return retval

	return inner


# Global Variables
# if os.environ.get('USER') == 'rjoshi2':
# 	PROJ_DIR	= '/projects/tir5/users/rjoshi2/medical_entity_linker/med_ent'
# 	WIKI_PATH	= '/projects/tir5/users/rjoshi2/medical_entity_linker/wikipedia/wikiextractor'
# 	DATA_PATH	= '/projects/tir5/users/rjoshi2/medical_entity_linker/med_ent'
# 	UMLS_DIR	= '/projects/tir5/users/rjoshi2/medical_entity_linker/umls/2019AB/META/'
# 	PUBMED_PATH     = '/projects/tir5/users/rjoshi2/medical_entity_linker/med_ent/pubmed'

# elif os.environ.get('USER') == 'shikhar':
# 	PROJ_DIR	= '/home/shikhar/med_ent'
# 	PUBMED_PATH     = '/home/shikhar/med_ent/data/pubmed'

# else:
# 	PROJ_DIR	= '/projects/med_ent'
# 	WIKI_PATH	= '/data/med_ent/wikipedia'
# 	PUBMED_PATH	= '/data/med_ent/pubmed'
# 	DATA_PATH 	= '/data/med_ent'
# 	UMLS_DIR	= '/data/bionlp_resources/umls/2019AB/META/'

# WD_PATH	= '/data/bionlp_resources/wikidata'
# FB_PATH	= '/data/bionlp_resources/datasets/freebase'
# c_mongo	= MongoClient('mongodb://{}:{}/'.format('brandy.lti.cs.cmu.edu', 27017), username='vashishths', password='yogeshwara')

DATA_MAP = {
	'chemdner': {
		'train'	: '/projects/chem_nlp/data/chemdner_corpus/chemdner_training.ner.sen.token4.BIO_allfea',
		'valid'	: '/projects/chem_nlp/data/chemdner_corpus/chemdner_development.ner.sen.token4.BIO_allfea',
		'test'	: '/projects/chem_nlp/data/chemdner_corpus/chemdner_evaluation.ner.sen.token4.BIO_allfea'	
	}
}

def partition(lst, n):
	if n == 0: return lst
	division = len(lst) / float(n)
	return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def get_chunks(inp_list, chunk_size):
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))

def dump_pickle(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))
	print('Pickled Dumped {}'.format(fname))

def load_pickle(fname):
	return pickle.load(open(fname, 'rb'))

def dump_dill(obj, fname):
	dill.dump(obj, open(fname, 'wb'))
	print('Pickled Dumped {}'.format(fname))

def load_dill(fname):
	return dill.load(open(fname, 'rb'))

def make_dir(dirpath):
	if not os.path.exists(dirpath):
		os.makedirs(dirpath)

def make_file(fname, mode):
	dir_name  = '/'.join(fname.split('/')[:-1])
	if not os.path.exists(dir_name): os.system(f'mkdir -p {dir_name}')
	return open(fname, mode)

def check_file(filename):
	return pathlib.Path(filename).is_file()

def str_proc(x):
	return str(x).strip().lower()

def set_gpu(gpus):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_logger(name, log_dir, config_dir):
	config_dict = json.load(open('{}/log_config.json'.format(config_dir)))
	config_dict['handlers']['file_handler']['filename'] = '{}/{}'.format(log_dir, name.replace('/', '-'))
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def to_gpu(batch, dev):
	batch_gpu = {}
	for key, val in batch.items():
		if   key.startswith('_'):	batch_gpu[key] = val
		elif type(val) == type({1:1}): 	batch_gpu[key] = {k: v.to(dev) for k, v in batch[key].items()}
		else: 				batch_gpu[key] = val.to(dev)
	return batch_gpu


class ResultsMongo:
	def __init__(self, params, ip='brandy.lti.cs.cmu.edu', port=27017, db='chem_nlp', username='vashishths', password='yogeshwara'):
		self.p		= params
		self.client	= MongoClient('mongodb://{}:{}/'.format(ip, port), username=username, password=password)
		self.db		= self.client[db][self.p.log_db]
		# self.db.update_one({'_id': self.p.name}, {'$set': {'Params': }}, upsert=True)

	def add_results(self, best_val, best_test, best_epoch, train_loss):
		try:
			self.db.update_one({'_id': self.p.name}, {
				'$set': {
					'best_epoch'	: best_epoch,
					'best_val'	: best_val,
					'best_test'	: best_test,
					'Params'	: vars(self.p)
				},
				'$push':{
					'train_loss'	: round(float(train_loss), 4),
					'all_val'	: best_val,
					'all_test'	: best_test,
				}
			}, upsert=True)
		except Exception as e:
			print('\nMongo Exception Cause: {}'.format(e.args[0]))

	def add_results_eval(self, res):
		try:
			self.db.update_one({'_id': self.p.name}, {
				'$set': {
					'partial_full'	: {
						'prec'	: res[-3],
						'rec'	: res[-2],
						'f1'	: res[-1],
					},
					'strict_full'	: {
						'prec'	: res[-6],
						'rec'	: res[-5],
						'f1'	: res[-4],
					},
					'kg_id'	: {
						'prec'	: res[-9],
						'rec'	: res[-8],
						'f1'	: res[-7],
					},
					'partial_men'	: {
						'prec'	: res[-12],
						'rec'	: res[-11],
						'f1'	: res[-10],
					},
					'strict_men'	: {
						'prec'	: res[-15],
						'rec'	: res[-14],
						'f1'	: res[-13],
					},
					'Params'	: vars(self.p)
				}
			}, upsert=True)
		except Exception as e:
			print('\nMongo Exception Cause: {}'.format(e.args[0]))

def read_csv(fname):
	with open(fname) as f:
		f.readline()
		for data in csv.reader(f):
			yield data

def mean_dict(acc):
	return {k: np.round(np.mean(v), 3) for k, v in acc.items()}

def comb_dict(res):
	return {k: [x[k] for x in res] for k in res[0].keys()}

def get_param(shape):
	param = Parameter(torch.Tensor(*shape)); 	
	xavier_normal_(param.data)
	return param

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def tune_thresholds(labels, logits, method = 'tune'):
	'''
	Takes labels and logits and tunes the thresholds using two methods
	methods are 'tune' or 'zc' #Zach Lipton
	Returns a list of tuples (thresh, f1) for each feature
	'''
	if method not in ['tune', 'zc']:
		print ('Tune method should be either tune or zc')
		sys.exit(1)
	
	res = []
	logits = sigmoid(logits)

	num_labels = labels.shape[1]

	def tune_th(pid, feat):
		max_f1, th = 0, 0		# max f1 and its thresh
		if method == 'tune':
			ts_to_test = np.arange(0, 1, 0.001)
			for t in ts_to_test:
				scr  = f1_score(labels[:, feat], logits[:, feat] > t)
				if scr > max_f1:
					max_f1	= scr
					th	= t
		else:
			f1_half = f1_score(labels[:, feat], logits[:, feat] > 0.5)
			th = f1_half / 2
			max_f1 = f1_score(labels[:, feat], logits[:, feat] > th)

		return (th, max_f1)
		
	res = Parallel(n_jobs = 1)(delayed(tune_th)(lbl, lbl) for lbl in range(num_labels))
	return res

def clean_text(text):
	text = str(text.encode('ascii', 'replace').decode())
	text = text.replace('\n',' ')
	text = text.replace('\t',' ')
	text = text.replace('|',' ')
	text = text.replace('\'',' ')
	return text


def replace(s, ch): 
	new_str = [] 
	l = len(s) 
	  
	for i in range(len(s)): 
		if (s[i] == ch and i != (l-1) and
		   i != 0 and s[i + 1] != ch and s[i-1] != ch): 
			new_str.append(s[i]) 
			  
		elif s[i] == ch: 
			if ((i != (l-1) and s[i + 1] == ch) and
			   (i != 0 and s[i-1] != ch)): 
				new_str.append(s[i]) 
				  
		else: 
			new_str.append(s[i]) 
		  
	return ("".join(i for i in new_str))


def getGlove(wrd_list, embed_type, c_dosa=None, random_embed=False):
	dim	= int(embed_type.split('_')[1])
	embeds	= np.zeros((len(wrd_list), dim), np.float32)
	if random_embed:
		count = 0
		for wrd in wrd_list:
			embeds[count, :] = np.random.randn(dim)
			count += 1
	else:
		if c_dosa == None: c_dosa = MongoClient('mongodb://{}:{}/'.format('brandy.lti.cs.cmu.edu', 27017), username='vashishths', password='yogeshwara')
		db_glove = c_dosa['glove'][embed_type]

		embed_map = {}
		res = db_glove.find({"_id": {"$in": wrd_list}})
		for ele in res:
			embed_map[ele['_id']] = ele['vec']

		count = 0
		for wrd in wrd_list:
			if wrd in embed_map: 	embeds[count, :] = np.float32(embed_map[wrd])
			else: 			embeds[count, :] = np.random.randn(dim)
			count += 1

	return embeds

class COL:
	HEADER = '\033[95m' 	# Violet
	INFO = '\033[94m'	# Blue 
	SUCCESS = '\033[92m'	# Green
	WARNING = '\033[93m'	# Yellow
	FAIL = '\033[91m'	# Red

	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

def color(s, col):
	if   col == 'head': return f"{COL.HEADER}{s}{COL.ENDC}"
	elif col == 'info': return f"{COL.INFO}{s}{COL.ENDC}"
	elif col == 'warn': return f"{COL.WARNING}{s}{COL.ENDC}"
	elif col == 'succ': return f"{COL.SUCCESS}{s}{COL.ENDC}"
	elif col == 'fail': return f"{COL.FAIL}{s}{COL.ENDC}"
	elif col == 'def': return f"{s}"
	return s

def parse_ann(fname, sent_offset, lbl_remap={}):
	ann_data = ddict(list)

	for line in open(fname):

		index, info, word	= line.split('\t')
		label, start, end 	= [x for x in info.split() if ';' not in x]
		
		start, end		= int(start), int(end)
		sent_num 		= None								

		for i, (sent_start, sent_end) in enumerate(sent_offset):
			if start >= sent_start and end <= sent_end:
				sent_num = i
				break

		if sent_num is None: continue

		ann_data[sent_num].append({
			'_id'	: index, 
			'start'	: start,
			'end'	: end,
			'word'	: word,
			'label'	: lbl_remap.get(label, label),
		})

	return ann_data

def parse_gaz(fname, sent_offset):
	gaz_data = ddict(list)

	for line in open(fname):

		start,  end, word, ids	= line.split('\t')
		# label, start, end 	= [x for x in info.split() if ';' not in x]
		start, end		= int(start), int(end)
		sent_num 		= None								

		for i, (sent_start, sent_end) in enumerate(sent_offset):
			if start >= sent_start and end <= sent_end:
				sent_num = i
				break

		if sent_num is None: continue

		gaz_data[sent_num].append({
			'start'	: start,
			'end'	: end,
			'word'	: word,
			'label'	: 'GAZ'
		})

	return gaz_data

from stop_words import get_stop_words
from nltk.corpus import stopwords 
stop_words  = get_stop_words('english')
stop_words.extend(list(stopwords.words('english')))
stop_words  = set(stop_words)

def check_stopword(word):
	if word.lower() in stop_words:
		return True
	return False

def combine_gen(iters, data_idx=0):
	count    = 0
	kill_cnt = 0
	iters 	 = [x.__iter__() for x in iters]

	while True:
		count += 1
		idx    = count % len(iters)

		try: 			ele = iters[idx].__next__()
		except Exception as e: 	ele = None
		
		if idx == data_idx and ele is None: return

		if ele is None:
			kill_cnt += 1
			if kill_cnt == len(iters):
				return
			continue
		else:
			kill_cnt = 0
			yield ele

def sizeof_fmt(num, suffix='B'):
	''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
	for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
		if abs(num) < 1024.0:
			return "%3.1f %s%s" % (num, unit, suffix)
		num /= 1024.0
	return "%.1f %s%s" % (num, 'Yi', suffix)
