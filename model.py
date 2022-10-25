import torch
from Transformer import MultiHeadAttention, PositionwiseFeedForward, SelfAttentionBlock, TransformerLayer

from layers import *
import torch.nn.functional as F
import torch.nn.init as torch_init


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)


class CMA_LA(nn.Module):
    def __init__(self, modal_a, modal_b, hid_dim=128, d_ff=512, dropout_rate=0.1):
        super(CMA_LA, self).__init__()

        self.cross_attention = CrossAttention(modal_b, modal_a, hid_dim)
        self.ffn = nn.Sequential(
            nn.Conv1d(modal_a, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(d_ff, 128, kernel_size=1),
            nn.Dropout(dropout_rate),
        )
        self.norm = nn.LayerNorm(modal_a)

    def forward(self, x, y, adj):
        new_x = x + self.cross_attention(y, x, adj)
        new_x = self.norm(new_x)
        new_x = new_x.permute(0, 2, 1)
        new_x = self.ffn(new_x)

        return new_x


# class Model(nn.Module):
#     def __init__(self, args):
#         super(Model, self).__init__()

#         n_features = args.feature_size
#         n_class = args.num_classes
#         self.dis_adj = DistanceAdj()

#         self.cross_attention = CMA_LA(modal_a=1024, modal_b=128, hid_dim=128, d_ff=512)
#         self.classifier = nn.Conv1d(128, 1, 7, padding=0)
#         self.apply(weight_init)

#     def forward(self, x):
#         f_v = x[:, :, :1024]  # (b, n, 1024)
#         f_a = x[:, :, 1024:]  # (b, n, 128)
#         adj = self.dis_adj(f_v.shape[0], f_v.shape[1])  # (b, n, n)

#         new_v = self.cross_attention(f_v, f_a, adj)  # (b, 128, n)
#         new_v = F.pad(new_v, (6, 0))  # padding for conv1d, (b, 128, 6 + n)
#         logits = self.classifier(new_v)  # (b, 1, n)
#         logits = logits.squeeze(dim=1)
#         logits = torch.sigmoid(logits)  # (b, n)

#         return logits


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        n_features = args.feature_size
        n_class = args.num_classes

        hid_dim = 128
        ffn_dim = hid_dim
        nhead = 4
        dropout = 0.1

        # self.fc_v = nn.Linear(1024, hid_dim)
        # self.fc_a = nn.Linear(128, hid_dim)
        self.fc = nn.Linear(1024 + 128, hid_dim)

        # self.self_attention_v = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))
        # self.self_attention_a = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))
        self.self_attention = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))

        # self.classifier_v = nn.Conv1d(hid_dim, 1, 7, padding=0)
        # self.classifier_a = nn.Conv1d(hid_dim, 1, 7, padding=0)
        self.classifier = nn.Conv1d(hid_dim, 1, 7, padding=0)

        self.apply(weight_init)

    def forward(self, x):
        f_v = x[:, :, :1024]  # (b, n, 1024)
        f_a = x[:, :, 1024:]  # (b, n, 128)

        f = self.fc(x)

        new_f = self.self_attention(f).permute(0, 2, 1)
        new_f = F.pad(new_f, (6, 0))
        # new_v = self.self_attention_v(f_v).permute(0, 2, 1)  # (b, 128, n)
        # new_v = F.pad(new_v, (6, 0))  # padding for conv1d, (b, 128, 6 + n)
        # new_a = self.self_attention_a(f_a).permute(0, 2, 1)  # (b, 128, n)
        # new_a = F.pad(new_a, (6, 0))  # padding for conv1d, (b, 128, 6 + n)
        logits = self.classifier(new_f).squeeze(1)
        # logits_v = self.classifier_v(new_v).squeeze(1)  # (b, 1, n) 
        # logits_a = self.classifier_a(new_a).squeeze(1)  # (b, 1, n)
        logits = torch.sigmoid(logits)
        # logits_v = torch.sigmoid(logits_v)  # (b, n)
        # logits_a = torch.sigmoid(logits_a)  # (b, n)

        return logits
