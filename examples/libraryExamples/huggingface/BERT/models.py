import torch
import torch.nn as nn
from transformers import modeling_outputs

# =============================================================================
# Model Wrappers for PAI
# =============================================================================
class RobertaForSequenceClassificationPB(nn.Module):
    def __init__(self, hf_model, dropout=0.1, dsn=False):
        super().__init__()
        # Copy submodules from the Hugging Face model.
        self.roberta = hf_model.roberta
        self.classifier = hf_model.classifier
        self.config = hf_model.config
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(dropout)
        
        # If the number of encoder layers is 0 (DSN mode), remove encoder layers and pooler.
        self.dsn = dsn
        if self.dsn:
            self.roberta.encoder.layer = nn.ModuleList([])
            if hasattr(self.roberta, 'pooler'):
                self.roberta.pooler = None

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Remove any extraneous arguments
        kwargs.pop("num_items_in_batch", None)
        
        if not self.dsn:
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                **kwargs
            )
            sequence_output = outputs[0]
        else:
            ## Deep Averaging Network - no encoder layers
            outputs = self.roberta.embeddings(
                input_ids=input_ids,
                position_ids=None,
                token_type_ids=token_type_ids,
                inputs_embeds=None,
                past_key_values_length=0,
            )
            # Average over the sequence dim
            sequence_output = torch.sum(outputs, dim=1, keepdim=True)
            
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )


class BertForSequenceClassificationPB(nn.Module):
    def __init__(self, hf_model, dropout=0.1, dsn=False):
        super().__init__()
        # Copy submodules from the Hugging Face model
        self.bert = hf_model.bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = hf_model.classifier
        self.config = hf_model.config
        self.num_labels = hf_model.config.num_labels
        
        self.dsn = dsn
        if self.dsn:
            self.bert.encoder.layer = nn.ModuleList([])
            # Remove the bert.pooler entirely
            self.bert.pooler = None
            
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Remove any extraneous arguments
        kwargs.pop("num_items_in_batch", None)
        
        if not self.dsn:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                **kwargs
            )
            pooled_output = outputs[1]
        else:
            ## Deep Averaging Network - no encoder layers
            outputs = self.bert.embeddings(
                input_ids=input_ids,
                position_ids=None,
                token_type_ids=token_type_ids,
                inputs_embeds=None,
                past_key_values_length=0,
            )
            # Average over the sequence dim
            pooled_output = torch.sum(outputs, dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )


# A simple wrapper for classifiers (if needed for PAI conversion)
class ClassifierWrapper(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(x)
