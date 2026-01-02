"""
DendriticBERT Core Implementation.

This module provides the DendriticBERT model architecture, which leverages
dendritic optimization for parameter-efficient NLP tasks.
"""

import torch
import torch.nn as nn
from transformers import modeling_outputs
from perforatedai import utils_perforatedai as upa


class DendriticBERT(nn.Module):
    """
    DendriticBERT: A parameter-efficient BERT model using Dendritic Optimization.
    """

    def __init__(self, hf_model, dropout=0.1, dsn=True, task_type="classification"):
        """
        Initialize the DendriticBERT model.

        Args:
            hf_model: The base Hugging Face model.
            dropout: Dropout probability.
            dsn: Whether to enable DSN mode (remove encoder layers).
            task_type: Type of task ("classification" or "qa").
        """
        super().__init__()
        self.bert = hf_model.bert
        self.dropout = nn.Dropout(dropout)
        self.config = hf_model.config
        self.task_type = task_type

        if self.task_type == "classification":
            self.classifier = hf_model.classifier
            self.num_labels = hf_model.config.num_labels
        elif self.task_type == "qa":
            self.qa_outputs = hf_model.qa_outputs

        self.dsn = dsn
        if self.dsn:
            self.bert.encoder.layer = nn.ModuleList([])
            self.bert.pooler = None
            print(f"DSN Mode ({task_type}): Encoder layers removed.")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        start_positions=None,
        end_positions=None,
        **kwargs,
    ):
        """
        Forward pass of the model.
        """
        kwargs.pop("num_items_in_batch", None)

        if not self.dsn:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                **kwargs,
            )
            sequence_output = outputs[0]
            pooled_output = outputs[1]
        else:
            outputs = self.bert.embeddings(
                input_ids=input_ids,
                position_ids=None,
                token_type_ids=token_type_ids,
                inputs_embeds=None,
                past_key_values_length=0,
            )
            sequence_output = outputs
            pooled_output = torch.mean(outputs, dim=1)

        if self.task_type == "classification":
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return modeling_outputs.SequenceClassifierOutput(
                loss=loss, logits=logits
            )

        elif self.task_type == "qa":
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            total_loss = None
            if start_positions is not None and end_positions is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2

            return modeling_outputs.QuestionAnsweringModelOutput(
                loss=total_loss,
                start_logits=start_logits,
                end_logits=end_logits,
            )


def initialize_dendritic_model(model):
    """
    Initialize the model with PerforatedAI's dendritic optimization.
    """
    print("Initializing Dendritic Optimization...")
    model = upa.initialize_pai(model)
    return model
