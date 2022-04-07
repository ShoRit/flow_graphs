from helper import *
from dataloader import *
from preprocess import *
from model import ZSBert, ZSBert_RGCN
from torch.utils.data import DataLoader
from zsbert_evaluate import extract_relation_emb, evaluate
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

def seed_everything():
	SEED = args.seed
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True
	os.environ['CUDA_VISIBLE_DEVICES']= args.gpu
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	return device


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--src_dataset",    help="choice of src_dataset",       type=str, default='risec')
	parser.add_argument("--tgt_dataset",    help="choice of tgt_dataset",       type=str, default='japflow')
	parser.add_argument("--mode",      		help="choice of operation",       	type=str, default='eval')
	parser.add_argument("--domain",      	help="seen or unseen domain",      	type=str, default='src') # src -->

	parser.add_argument("--bert_model",     help="choice of bert model  ",  	type=str, default='bert-base-uncased')
	parser.add_argument("--seed", 			help="random seed", 				type=int, default=300)
	parser.add_argument("--gpu", 			help="choice of device", 			type=str, default='0')
	
	parser.add_argument("--node_vec", 		help="number of unseen classes", 	type=int, default=100)
	parser.add_argument("--dep", 			help="dependency_parsing", 			type=str, default='1')
 
	parser.add_argument("--n_unseen", 		help="number of unseen classes", 	type=int, default=10)
	parser.add_argument("--gamma", 			help="margin factor gamma", 		type=float, default=7.5)
	parser.add_argument("--alpha", 			help="balance coefficient alpha", 	type=float, default=0.5)
	parser.add_argument("--dist_func", 		help="distance computing function", type=str, default='cosine')
	parser.add_argument("--batch_size", 										type=int, default=32)
	parser.add_argument("--epochs", 											type=int, default=50)
	parser.add_argument("--patience", 											type=int, default=5)
	parser.add_argument("--num_neighbors", 										type=int, default=2)
	parser.add_argument("--omit_rels", 											type=str, default='')


	args = parser.parse_args()

	return args


def get_lbl_features(data, rel2desc_emb):

	labels 				= [elem['label'] for elem in data]
	lbl2id 				= {}
	for elem in labels:
		if elem not in lbl2id: 
			lbl2id[elem] = len(lbl2id)

	test_y_attr, test_y = [], []
	test_idxmap 		= {}

	for i, elem in enumerate(data): 	
		lbl             = lbl2id[elem['label']]
		test_y.append(lbl)
		test_idxmap[i]  = lbl
	
	test_y_attr = [rel2desc_emb[i] for i in lbl2id]    
	test_y_attr = np.array(test_y_attr)
	test_y      = np.array(test_y)

	return test_y, test_idxmap, labels, test_y_attr, lbl2id


def get_known_lbl_features(data, rel2desc_emb, lbl2id):
	labels 				= [elem['label'] for elem in data]

	test_y_attr, test_y = [], []
	test_idxmap 		= {}

	for i, elem in enumerate(data): 	
		lbl             = lbl2id[elem['label']]
		test_y.append(lbl)
		test_idxmap[i]  = lbl
	
	test_y_attr = [rel2desc_emb[i] for i in lbl2id]    
	test_y_attr = np.array(test_y_attr)
	test_y      = np.array(test_y)

	return test_y, test_idxmap, labels, test_y_attr, lbl2id


# def extract_relation_emb(model, testloader, device):
#     out_relation_embs = None
#     model.eval()
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     for data in tqdm(testloader):
#         tokens_tensors, segments_tensors, marked_e1, marked_e2, masks_tensors, relation_emb, labels, graph_data = [t.to(device) for t in data if t is not None]

#         # tokens_tensors, segments_tensors, marked_e1, marked_e2, masks_tensors, relation_emb = [t.to(device) for t in data if t is not None]
#         with torch.no_grad():
#             outputs, out_relation_emb = model(input_ids=tokens_tensors, 
#                                         token_type_ids=segments_tensors,
#                                         e1_mask=marked_e1,
#                                         e2_mask=marked_e2,
#                                         attention_mask=masks_tensors,
#                                         input_relation_emb=relation_emb,
#                                         graph_data= graph_data,
# 										device =device)
#             logits = outputs[0]
#         if out_relation_embs is None:
#             out_relation_embs = out_relation_emb
#         else:
#             out_relation_embs = torch.cat((out_relation_embs, out_relation_emb))    
#     return out_relation_embs


def seen_eval(model, loader, device):
	model.eval()
	correct, total = 0, 0
	y_true, y_pred = [],[]
	for data in tqdm(loader):
		tokens_tensors, segments_tensors, marked_e1, marked_e2, masks_tensors, relation_emb, labels, graph_data = [t.to(device) for t in data]	
		
		with torch.no_grad():
			outputs, out_relation_emb = model(input_ids=tokens_tensors, 
										token_type_ids=segments_tensors,
										e1_mask=marked_e1,
										e2_mask=marked_e2,
										attention_mask=masks_tensors,
										input_relation_emb=relation_emb,
										labels=labels,
										graph_data= graph_data,
										device =device)
			logits = outputs[1]
   
		total += labels.size(0)
		_, pred = torch.max(logits, 1)
		correct += (pred == labels).sum().item()
		y_pred.extend(list(np.array(pred.cpu().detach())))
		y_true.extend(list(np.array(labels.cpu().detach())))

	f1		  							  = f1_score(y_true,y_pred, average="macro")
	p1, r1 								  = precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro')
	# print(f'val acc: {correct/total}, f1 : {f1_score(y_true,y_pred, average="macro")}')
 
	return p1, r1, f1
 

def main(args):
	device                          			=   seed_everything()
	tokenizer                       			=   BertTokenizerFast.from_pretrained(args.bert_model)
	src_dir 									= 	f'../data/{args.src_dataset}'
	src_file 									= 	f'{src_dir}/data.dill'
	
	# if check_file(src_file):
	# 	src_dataset								=   load_dill(src_file)
	# else:
	src_dataset, deprel_dict			        =   create_datafield(src_dir,['train','dev','test'],tokenizer)
  
	train_data, dev_data, test_data     		=   src_dataset['train']['rels'], src_dataset['dev']['rels'], src_dataset['test']['rels']
 
	# here we end up creating relations based on the description of the label completely, hence we need a unified form of labels to use for the seen case.
	
	rel2desc, all_rel2id, id2all_rel, rel2desc_emb = generate_reldesc()
	
	print('train size: {}, dev size {}, test size: {}'.format(len(train_data), len(dev_data), len(test_data)))
	print('Data is successfully loaded')
	train_label 								= [data['label'] for data in train_data]

	bertconfig = BertConfig.from_pretrained(args.bert_model,num_labels=len(set(train_label)))

	if  'bert-large' in args.bert_model: bertconfig.relation_emb_dim = 1024
	elif 'bert-base' in args.bert_model: bertconfig.relation_emb_dim = 768

	bertconfig.margin 		= args.gamma
	bertconfig.alpha 		= args.alpha
	bertconfig.dist_func 	= args.dist_func
 
	bertconfig.node_vec 	= args.node_vec
	bertconfig.dep_rels 	= len(deprel_dict)
	bertconfig.dep			= args.dep
	
	model 					= ZSBert_RGCN.from_pretrained(args.bert_model, config=bertconfig)
	# model 					= ZSBert_RGCN(config=bertconfig)
	model       			= model.to(device)

	if args.domain 								== 'src':
		labels 															  = sorted(list(set(train_label)))
		lbl2id 															  = {lbl:idx for idx, lbl in enumerate(labels)}
		train_y, train_idxmap, train_labels, train_y_attr, train_lbl2id   = get_known_lbl_features(train_data, rel2desc_emb, lbl2id)
	else:
		train_y, train_idxmap, train_labels, train_y_attr, train_lbl2id   = get_lbl_features(train_data, rel2desc_emb)
 
	trainset    = ZSBert_RGCN_RelDataset(train_data, train_lbl2id, tokenizer, args, domain='src')
	trainloader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=create_mini_batch, shuffle=True)
	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)

	best_p, best_r, best_f1 = 0, 0,0
	checkpoint_file													= f'../checkpoints/{args.src_dataset}_{args.domain}_alpha_{args.alpha}_gamma_{args.gamma}_dist_{args.dist_func}_dep_{args.dep}.pt'

	if args.mode 													== 'train':
		if args.domain 												=='src':
			dev_y, dev_idxmap, dev_labels, dev_y_attr, dev_lbl2id   = get_known_lbl_features(dev_data, rel2desc_emb, lbl2id)
		else:
			dev_y, dev_idxmap, dev_labels, dev_y_attr, dev_lbl2id   = get_lbl_features(dev_data, rel2desc_emb)

		devset        							   				= ZSBert_RGCN_RelDataset(dev_data, dev_lbl2id, tokenizer, args, domain=args.domain)
		devloader     							   				= DataLoader(devset, batch_size=args.batch_size, collate_fn=create_mini_batch)
		kill_cnt 												= 0
		for epoch in range(args.epochs):
			print(f'============== TRAIN ON THE {epoch+1}-th EPOCH ==============')
			running_loss, correct, total = 0.0, 0,0

			model.train()
			y_true, y_pred = [],[]

			for data in tqdm(trainloader):
       
				tokens_tensors, segments_tensors, marked_e1, marked_e2, masks_tensors, relation_emb, labels, graph_data = [t.to(device) for t in data]
				
				
				optimizer.zero_grad()
				outputs, out_relation_emb = model(input_ids=tokens_tensors, 
											token_type_ids=segments_tensors,
											e1_mask=marked_e1,
											e2_mask=marked_e2,
											attention_mask=masks_tensors,
											input_relation_emb=relation_emb,
											labels=labels,
											graph_data= graph_data,
											device =device)
				loss   = outputs[0]
				logits = outputs[1]
				total += labels.size(0)
				_, pred = torch.max(logits, 1)
				correct += (pred == labels).sum().item()
				loss.backward()
				optimizer.step()
				running_loss += loss.item()

				y_pred.extend(list(np.array(pred.cpu().detach())))
				y_true.extend(list(np.array(labels.cpu().detach())))

				# if step % 1000 == 0: print(f'[step {step}]' + '=' * (step//1000))

			print(f'train acc: {correct/total}, f1 : {f1_score(y_true,y_pred, average="macro")}')

			print('============== EVALUATION ON DEV DATA ==============')

			if args.domain == 'src': 
				pt, rt, f1t 	 = seen_eval(model, trainloader, device=device)
				print(f'Train data {f1t} \t Prec {pt} \t Rec {rt}')
				pt, rt, f1t 	 = seen_eval(model, devloader, device=device)
				print(f'Eval data {f1t} \t Prec {pt} \t Rec {rt}')
    
			else:
				preds        	 = extract_relation_emb(model, devloader, device=device).cpu().numpy()
				pt, rt, f1t,h_K  = evaluate(preds, dev_y_attr, dev_y, dev_idxmap, args.dist_func)
				print(f'[val] precision: {pt:.4f}, recall: {rt:.4f}, f1 score: {f1t:.4f}, H@K :{h_K}')

			if f1t > best_f1:
				best_p, best_r, best_f1 = pt, rt, f1t
				best_model = model
				kill_cnt = 0
				torch.save(best_model.state_dict(),checkpoint_file)
			else:
				kill_cnt +=1
				if kill_cnt >= args.patience:
					torch.save(best_model.state_dict(),checkpoint_file)
					break
			
			print(f'[best val] precision: {best_p:.4f}, recall: {best_r:.4f}, f1 score: {best_f1:.4f}')
		
		torch.save(best_model.state_dict(),checkpoint_file)

	if args.mode												 		== 'eval':
		if not check_file(checkpoint_file) :
			print('MODEL CANNOT BE LOADED BY THESE SPECIFICATIONS')
			return
		if args.domain 													==	'src':
			test_y, test_idxmap, test_labels, test_y_attr, test_lbl2id  = 	get_known_lbl_features(test_data, rel2desc_emb, lbl2id)
		else:
			test_y, test_idxmap, test_labels, test_y_attr, test_lbl2id  = 	get_lbl_features(test_data, rel2desc_emb)
   
		testset        							   					 	= 	ZSBert_RGCN_RelDataset(test_data, test_lbl2id, tokenizer, args)
		testloader     							   					 	= 	DataLoader(testset, batch_size=args.batch_size, collate_fn=create_mini_batch)
  
		model 															= 	ZSBert_RGCN.from_pretrained(args.bert_model, config=bertconfig)
		model       													= 	model.to(device)
		model.load_state_dict(torch.load(checkpoint_file))
		model.eval()
  
		if args.domain 	=='src':
			pt, rt, f1t 	 = seen_eval(model, testloader, device=device)
			print(f'Train data {f1t} \t Prec {pt} \t Rec {rt}')
		else:
			preds       												 = extract_relation_emb(model, testloader, device=device).cpu().numpy()
			pt, rt, f1t, h_K  											 = evaluate(preds, test_y_attr, test_y, test_idxmap, args.dist_func, test_lbl2id, num_neighbors=args.num_neighbors)

			print(f'[best test] precision: {pt:.4f}, recall: {rt:.4f}, f1 score: {f1t:.4f}, H@K : {h_K}')



if __name__ =='__main__':	
	args                            =   get_args()
	main(args)