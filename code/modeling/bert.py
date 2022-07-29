from abc import ABC

import numpy as np
import torch
from torch import nn as nn
from transformers import BertModel, BertPreTrainedModel

from modeling.graph_components import DeepNet


class BertRGCNRelationClassiferABC(BertPreTrainedModel, ABC):
    def __init__(self, config, use_graph_data):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.use_graph_data = use_graph_data
        if self.use_graph_data:
            self.gnn = DeepNet(config, config.dep_rels)

    def extract_entity(self, sequence_output, e_mask):
        extended_e_mask = e_mask.unsqueeze(1)
        extended_e_mask = torch.bmm(extended_e_mask.float(), sequence_output).squeeze(1)
        return extended_e_mask.float()

    def embed_context_and_entities(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        e1_mask=None,
        e2_mask=None,
        head_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
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

        return context, e1_h, e2_h, sequence_output

    def embed_entity_nodes(self, graph_data, sequence_output):
        n1_mask = graph_data.n1_mask
        n2_mask = graph_data.n2_mask
        batch = graph_data.batch

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

        e1_node_emb, e1_node_emb = [], []
        for idx in range(0, sequence_output.shape[0]):
            mask = torch.where(batch == idx, 1, 0)
            m1, m2 = mask * n1_mask, mask * n2_mask
            e1_node_emb.append(torch.mm(m1.unsqueeze(dim=0).float(), graph_embs))
            e1_node_emb.append(torch.mm(m2.unsqueeze(dim=0).float(), graph_embs))

        e1_node_emb = torch.cat(e1_node_emb, dim=0)
        e1_node_emb = torch.cat(e1_node_emb, dim=0)

        return e1_node_emb, e1_node_emb


class BertRGCNRelationClassifierConcat(BertRGCNRelationClassiferABC):
    def __init__(self, config, use_graph_data):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.relation_emb_dim = config.relation_emb_dim

        rel_in = config.hidden_size * 3
        self.use_graph_data = use_graph_data
        if self.use_graph_data:
            rel_in += config.node_emb_dim * 2

        self.fc_layer = nn.Linear(rel_in, self.relation_emb_dim)
        self.rel_classifier = nn.Linear(self.relation_emb_dim, self.config.num_labels)
        self.config = config

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
        labels=None,
        graph_data=None,
        **kwargs
    ):
        output_dict = {}

        e1_bert, e2_bert, context, sequence_output = self.embed_context_and_entities(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            e1_mask,
            e2_mask,
            head_mask,
            inputs_embeds,
        )
        rel_output = [context, e1_bert, e2_bert]

        if self.use_graph_data:
            e1_node, e2_node = self.embed_entity_nodes(graph_data, sequence_output)
            rel_output.extend([e1_node, e2_node])

        pooled_output = torch.cat(rel_output, dim=-1)

        pooled_output = torch.tanh(pooled_output)
        pooled_output = self.fc_layer(pooled_output)
        relation_embeddings = torch.tanh(pooled_output)

        output_dict["hidden_states"] = sequence_output
        output_dict["relation_embeddings"] = relation_embeddings

        if labels is not None:
            logits = self.rel_classifier(relation_embeddings)
            output_dict["logits"] = logits

            ce_loss = nn.CrossEntropyLoss()
            loss = ce_loss(logits.view(-1, self.num_labels), labels.view(-1))
            output_dict["loss"] = loss

        return output_dict


# shim in place while we configure the refactor
BertRGCNRelationClassifier = BertRGCNRelationClassifierConcat
