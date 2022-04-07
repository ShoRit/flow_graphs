import imp
import torch
import random
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from torch_geometric.nn import FastRGCNConv, RGCNConv


class ZSBert(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.relation_emb_dim = config.relation_emb_dim
		self.margin = torch.tensor(config.margin)
		self.alpha = config.alpha
		self.dist_func = config.dist_func
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.fclayer = nn.Linear(config.hidden_size*3, self.relation_emb_dim)
		self.classifier = nn.Linear(self.relation_emb_dim, self.config.num_labels)
		self.init_weights()

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		e1_mask=None,
		e2_mask=None,
		head_mask=None,
		inputs_embeds=None,
		input_relation_emb=None,
		labels=None,
		device='cpu'
	):
		
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)

		sequence_output = outputs[0] # Sequence of hidden-states of the last layer.
		pooled_output   = outputs[1] # Last layer hidden-state of the [CLS] token further processed 
									 # by a Linear layer and a Tanh activation function.

		def extract_entity(sequence_output, e_mask):
			extended_e_mask = e_mask.unsqueeze(1)
			extended_e_mask = torch.bmm(
				extended_e_mask.float(), sequence_output).squeeze(1)
			return extended_e_mask.float()
		
		e1_h = extract_entity(sequence_output, e1_mask)
		e2_h = extract_entity(sequence_output, e2_mask)
		context = self.dropout(pooled_output)
		pooled_output = torch.cat([context, e1_h, e2_h], dim=-1)
		pooled_output = torch.tanh(pooled_output)
		pooled_output = self.fclayer(pooled_output)
		relation_embeddings = torch.tanh(pooled_output)
		# relation_embeddings = self.dropout(relation_embeddings)
		logits = self.classifier(relation_embeddings) # [batch_size x hidden_size]
		outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

		if labels is not None:
			# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			gamma = self.margin.to(device)
			ce_loss = nn.CrossEntropyLoss()
			loss = (ce_loss(logits.view(-1, self.num_labels), labels.view(-1))) * self.alpha
			zeros = torch.tensor(0.).to(device)
			for a, b in enumerate(relation_embeddings):
				max_val = torch.tensor(0.).to(device)
				for i, j in enumerate(input_relation_emb):
					if a==i:
						if self.dist_func == 'inner':
							pos = torch.dot(b, j).to(device)
						elif self.dist_func == 'euclidian':
							pos = torch.dist(b, j, 2).to(device)
						elif self.dist_func == 'cosine':
							pos = torch.cosine_similarity(b, j, dim=0).to(device)
					else:
						if self.dist_func == 'inner':
							tmp = torch.dot(b, j).to(device)
						elif self.dist_func == 'euclidian':
							tmp = torch.dist(b, j, 2).to(device)
						elif self.dist_func == 'cosine':
							tmp = torch.cosine_similarity(b, j, dim=0).to(device)
						if tmp > max_val:
							if labels[a] != labels[i]:
								max_val = tmp
							else:
								continue
				neg = max_val.to(device)
#                 print(f'neg={neg}')
#                 print(f'neg-pos+gamma={neg - pos + gamma}')
#                 print('===============')
				if self.dist_func == 'inner' or self.dist_func == 'cosine':
					loss += (torch.max(zeros, neg - pos + gamma) * (1-self.alpha))
				elif self.dist_func == 'euclidian':
					loss += (torch.max(zeros, pos - neg + gamma) * (1-self.alpha))
			outputs = (loss,) + outputs

		return outputs, relation_embeddings  # (loss), logits, (hidden_states), (attentions)
	

class Net(torch.nn.Module):
	def __init__(self, num_node_features, num_relations):
		super().__init__()
		self.conv1 = RGCNConv(in_channels=num_node_features, out_channels=num_node_features, num_relations=num_relations)

	def forward(self, x, edge_index, edge_type):
		x = F.relu(self.conv1(x, edge_index, edge_type))
		return x

# class Net(torch.nn.Module):
#     def __init__(self, num_node_features, num_relations):
#         super().__init__()
#         self.conv1 = RGCNConv(in_channels=num_node_features, out_channels=num_node_features, num_relations=num_relations)
#     def forward(self, x, edge_index, edge_type):
#         x = F.relu(self.conv1(x, edge_index, edge_type))
#         return x

class ZSBert_RGCN(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.relation_emb_dim = config.relation_emb_dim
		self.margin = torch.tensor(config.margin)
		self.alpha = config.alpha
		self.dist_func = config.dist_func
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		if self.config.dep =='1':
			self.rgcn    = Net(config.node_vec, config.dep_rels)
			self.fclayer = nn.Linear(config.hidden_size*3 + config.node_vec*2, self.relation_emb_dim)
		else:
			self.fclayer = nn.Linear(config.hidden_size*3, self.relation_emb_dim)
			
		self.classifier = nn.Linear(self.relation_emb_dim, self.config.num_labels)
		self.config     = config
		self.init_weights()

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		e1_mask=None,
		e2_mask=None,
		head_mask=None,
		inputs_embeds=None,
		input_relation_emb=None,
		labels=None,
		graph_data = None,
		device='cpu'
	):
		# import pdb; pdb.set_trace()    
		
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)

		sequence_output = outputs[0] # Sequence of hidden-states of the last layer.
		pooled_output   = outputs[1] # Last layer hidden-state of the [CLS] token further processed 
									 # by a Linear layer and a Tanh activation function.

		def extract_entity(sequence_output, e_mask):
			extended_e_mask = e_mask.unsqueeze(1)
			extended_e_mask = torch.bmm(
				extended_e_mask.float(), sequence_output).squeeze(1)
			return extended_e_mask.float()
		
											 
		e1_h = extract_entity(sequence_output, e1_mask)
		e2_h = extract_entity(sequence_output, e2_mask)
		context = self.dropout(pooled_output)
		
		if self.config.dep == '1':                                                                                                     
			graph_embs                  = self.rgcn(graph_data.x, graph_data.edge_index, graph_data.edge_type)
			
			n1_mask, n2_mask, batch     = graph_data.n1_mask, graph_data.n2_mask, graph_data.batch
			
			e1_dep, e2_dep              = [],[]
			for idx in range(0,sequence_output.shape[0]):
				mask        = torch.where(batch==idx, 1,0)
				m1, m2      = mask*n1_mask, mask*n2_mask
				e1_dep.append(torch.mm(m1.unsqueeze(dim=0).float(),graph_embs))
				e2_dep.append(torch.mm(m2.unsqueeze(dim=0).float(),graph_embs))
			
			e1_dep          = torch.cat(e1_dep, dim=0)
			e2_dep          = torch.cat(e2_dep, dim=0)

			pooled_output = torch.cat([context, e1_h, e2_h, e1_dep, e2_dep], dim=-1)

		else:
			pooled_output = torch.cat([context, e1_h, e2_h], dim=-1)
			
		pooled_output = torch.tanh(pooled_output)
		pooled_output = self.fclayer(pooled_output)
		relation_embeddings = torch.tanh(pooled_output)
		# relation_embeddings = self.dropout(relation_embeddings)
		logits = self.classifier(relation_embeddings) # [batch_size x hidden_size]
		outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

		if labels is not None:
			# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			gamma = self.margin.to(device)
			ce_loss = nn.CrossEntropyLoss()
			loss = (ce_loss(logits.view(-1, self.num_labels), labels.view(-1))) * self.alpha
			zeros = torch.tensor(0.).to(device)
			for a, b in enumerate(relation_embeddings):
				max_val = torch.tensor(0.).to(device)
				for i, j in enumerate(input_relation_emb):
					if a==i:
						if self.dist_func == 'inner':
							pos = torch.dot(b, j).to(device)
						elif self.dist_func == 'euclidian':
							pos = torch.dist(b, j, 2).to(device)
						elif self.dist_func == 'cosine':
							pos = torch.cosine_similarity(b, j, dim=0).to(device)
					else:
						if self.dist_func == 'inner':
							tmp = torch.dot(b, j).to(device)
						elif self.dist_func == 'euclidian':
							tmp = torch.dist(b, j, 2).to(device)
						elif self.dist_func == 'cosine':
							tmp = torch.cosine_similarity(b, j, dim=0).to(device)
						if tmp > max_val:
							if labels[a] != labels[i]:
								max_val = tmp
							else:
								continue
				neg = max_val.to(device)
#                 print(f'neg={neg}')
#                 print(f'neg-pos+gamma={neg - pos + gamma}')
#                 print('===============')
				if self.dist_func == 'inner' or self.dist_func == 'cosine':
					loss += (torch.max(zeros, neg - pos + gamma) * (1-self.alpha))
				elif self.dist_func == 'euclidian':
					loss += (torch.max(zeros, pos - neg + gamma) * (1-self.alpha))
			outputs = (loss,) + outputs

		return outputs, relation_embeddings  # (loss), logits, (hidden_states), (attentions)
	

'''
class BertRelClass(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config 	= config
		self.num_labels = config.num_labels
		self.bert		= BertModel(config)
		self.dropout	= nn.Dropout(self.p.drop)
		fc_in			= self.bert.config.hidden_size * 2
		self.classifier	= nn.Linear(fc_in, self.num_labels)

		self.bert 		= BertModel(config)
		self.dropout 	= nn.Dropout(config.hidden_dropout_prob)
		if self.config.dep =='1':
			self.rgcn    = Net(config.node_vec, config.dep_rels)
			self.fclayer = nn.Linear(config.hidden_size*3 + config.node_vec*2, self.relation_emb_dim)
		else:
			self.fclayer = nn.Linear(config.hidden_size*3, self.relation_emb_dim)
			
		self.classifier = nn.Linear(self.relation_emb_dim, self.config.num_labels)
		self.init_weights()

	
	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, e1_mask=None, e2_mask=None, \
					head_mask=None,inputs_embeds=None,input_relation_emb=None,labels=None, graph_data = None, device='cpu'):
     
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)		
  		
		sequence_output 	= outputs[0] # Sequence of hidden-states of the last layer.
		pooled_output   	= outputs[1] # Last layer hidden-state of the [CLS] token f


		arg1_emb			= torch.matmul(e1_mask.unsqueeze(dim=1),sequence_output).squeeze(dim=1)
		arg2_emb			= torch.matmul(e2_mask.unsqueeze(dim=1),sequence_output).squeeze(dim=1)
		arg1_sum			= sum(e1_mask.transpose(0,1))+1e-5
		arg2_sum			= sum(e1_mask.transpose(0,1))+1e-5
  
		arg1_emb 			= arg1_emb/arg1_sum.view(len(arg1_sum),1)
		arg2_emb			= arg2_emb/arg2_sum.view(len(arg2_sum),1)

		arg_emb 			= torch.cat((arg1_emb,arg2_emb), dim =1)


		logits 				= self.classifier(arg_emb)

		if self.p.wgh_loss:
			loss 				= F.binary_cross_entropy_with_logits(logits, bat['labels'].float(), weight = self.weights, reduction='sum')
		else:
			loss 				= F.binary_cross_entropy_with_logits(logits, bat['labels'].float(), reduction='sum')

		loss_val 			= loss.item()

		if loss_val != loss_val: import pdb; pdb.set_trace()

		return loss, logits

'''