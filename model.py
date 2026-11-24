import torch
import torch.nn as nn
from transformers import BertModel

class EmotionModel(nn.Module):

    def __init__(
        self,
        sent_num=3,
        emo_num=13,
        da_num=19
    ):

        super().__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-chinese"
        )

        hidden=768

        self.sent_classifier=nn.Linear(
            hidden,
            sent_num
        )

        self.emo_classifier=nn.Linear(
            hidden,
            emo_num
        )

        self.da_classifier=nn.Linear(
            hidden,
            da_num
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        output=self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled=output.pooler_output

        sent=self.sent_classifier(
            pooled
        )

        emo=self.emo_classifier(
            pooled
        )

        da=self.da_classifier(
            pooled
        )

        return sent,emo,da