import sys; sys.path.append('common')
from helper import *
from torchcrf import CRF

class BertPlain(nn.Module):
	def __init__(self, params, num_labels):
		super().__init__()
		
		self.p		= params
		self.bert	= AutoModel.from_pretrained(self.p.bert_dir)
		self.dropout	= nn.Dropout(self.p.drop)

		if self.p.crf: 
			self.crf = CRF(num_labels, batch_first=True)

		if self.p.lstm:
			self.lstm = nn.LSTM(self.bert.config.hidden_size, self.p.rnn_dim // 2, num_layers=self.p.rnn_layers, bidirectional=True, dropout=self.p.drop)
			fc_in 	  = self.p.rnn_dim
		else:   
			fc_in 	  = self.bert.config.hidden_size

		if self.p.gaz:
			fc_in     += 3

		self.classifier	= nn.Linear(fc_in, num_labels)
	
	def forward(self, bat):
		
		tok_embed = self.bert(
			input_ids 	= bat['tok_pad'],
			attention_mask	= bat['tok_mask']
		)[0]
		tok_embed = tok_embed[:, 1:-1, :]	# Removing embedding corresponding to [CLS] and [SEP]

		if self.p.lstm:
			packed		= pack_padded_sequence(tok_embed, bat['tok_len'].cpu(), batch_first=True)			
			rnn_out, _	= self.lstm(packed)
			fc_in, _	= pad_packed_sequence(rnn_out, batch_first=True)
		else:
			fc_in 		= tok_embed

		if self.p.gaz:
			fc_in       = torch.cat((fc_in, bat['gaztags']),dim=2)

		logits = self.classifier(fc_in)

		if self.p.crf:
			try:
				output 	= self.crf.decode(logits, mask=bat['val_pad'].bool())
				loss 	= -self.crf(logits, bat['labels'].argmax(2), mask=bat['val_pad'].bool())			
			except Exception as e:
				loss = None
		else:
			_loss	= F.binary_cross_entropy_with_logits(logits, bat['labels'].float(), reduction='none')
			loss	= (_loss * bat['val_pad'].unsqueeze(2)).sum() / (bat['val_pad'].sum() + 0.0000001)

		return loss, logits, tok_embed