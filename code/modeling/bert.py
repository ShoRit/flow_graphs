import numpy as np
import torch
from torch import nn as nn
from transformers import BertModel, BertPreTrainedModel

from modeling.graph_components import DeepNet


class BertRGCNRelationClassifier(BertPreTrainedModel):
    def __init__(self, config, use_graph_data):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.relation_emb_dim = config.relation_emb_dim
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        rel_in = config.hidden_size * 3
        self.use_graph_data = use_graph_data
        if self.use_graph_data:
            self.gnn = DeepNet(config, config.dep_rels)
            rel_in += config.node_emb_dim * 2

        self.fc_layer = nn.Linear(rel_in, self.relation_emb_dim)
        self.rel_classifier = nn.Linear(self.relation_emb_dim, self.config.num_labels)
        self.config = config

        self.init_weights()

    def extract_entity(self, sequence_output, e_mask):
        extended_e_mask = e_mask.unsqueeze(1)
        extended_e_mask = torch.bmm(extended_e_mask.float(), sequence_output).squeeze(1)
        return extended_e_mask.float()

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
        labels=None,
        graph_data=None,
        **kwargs
    ):
        output_dict = {}

        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = bert_outputs["last_hidden_state"]  # Sequence of hidden-states of the last
        # layer.
        pooled_output = bert_outputs["pooler_output"]  # Last layer hidden-state of the [CLS] token,
        # further processed by a Linear layer and a Tanh activation function.
        context = self.dropout(pooled_output)

        e1_h = self.extract_entity(sequence_output, e1_mask)
        e2_h = self.extract_entity(sequence_output, e2_mask)

        rel_output = [context, e1_h, e2_h]

        if self.use_graph_data:
            n1_mask, n2_mask, batch = (
                graph_data.n1_mask,
                graph_data.n2_mask,
                graph_data.batch,
            )
            batch_np = batch.cpu().detach().numpy()
            graph_embs = []
            for idx in range(0, sequence_output.shape[0]):
                bids = np.where(batch_np == idx)[0]
                sid, eid = bids[0], bids[-1] + 1
                graph_embs.append(
                    torch.max(sequence_output[idx] + graph_data.x[sid:eid, :, None], dim=1)[0]
                )

            graph_embs = torch.vstack(graph_embs)
            graph_embs[graph_embs == -torch.inf] = 0

            graph_embs = self.gnn(graph_embs, graph_data.edge_index, graph_data.edge_type)

            e1_dep, e2_dep = [], []
            for idx in range(0, sequence_output.shape[0]):
                mask = torch.where(batch == idx, 1, 0)
                m1, m2 = mask * n1_mask, mask * n2_mask
                e1_dep.append(torch.mm(m1.unsqueeze(dim=0).float(), graph_embs))
                e2_dep.append(torch.mm(m2.unsqueeze(dim=0).float(), graph_embs))

            e1_dep = torch.cat(e1_dep, dim=0)
            e2_dep = torch.cat(e2_dep, dim=0)

            rel_output.append(e1_dep)
            rel_output.append(e2_dep)

        pooled_output = torch.cat(rel_output, dim=-1)

        pooled_output = torch.tanh(pooled_output)
        pooled_output = self.fc_layer(pooled_output)
        relation_embeddings = torch.tanh(pooled_output)
        # relation_embeddings = self.dropout(relation_embeddings)
        # [batch_size x hidden_size]

        output_dict["hidden_states"] = bert_outputs[2:]
        output_dict["relation_embeddings"] = relation_embeddings

        if labels is not None:
            logits = self.rel_classifier(relation_embeddings)
            output_dict["logits"] = logits

            ce_loss = nn.CrossEntropyLoss()
            loss = ce_loss(logits.view(-1, self.num_labels), labels.view(-1))
            output_dict["loss"] = loss

        return output_dict
