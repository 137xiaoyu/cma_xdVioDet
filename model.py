import torch
from Transformer import CrossAttentionBlock, MultiHeadAttention, PositionwiseFeedForward, SelfAttentionBlock, TransformerLayer

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




class TemporalConsensus(nn.Module):
    def __init__(self, len_feature, hid_dim):
        super().__init__()

        nhead = 4
        dropout = 0.1
        ffn_dim = hid_dim
        bn = nn.BatchNorm1d
        self.hid_dim = hid_dim
        self.len_feature = len_feature

        self.fc_v = nn.Linear(self.len_feature, hid_dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim))
        self.cma = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))
        # self.mlp_head = nn.Sequential(nn.LayerNorm(hid_dim), nn.Linear(hid_dim, 1))

        self.fc_v2 = nn.Linear(self.len_feature, hid_dim)
        self.conv_1 = nn.Sequential(nn.Conv1d(in_channels=self.hid_dim, out_channels=self.hid_dim, kernel_size=3, stride=1, dilation=1, padding=1), bn(self.hid_dim), nn.ReLU())
        self.conv_2 = nn.Sequential(nn.Conv1d(in_channels=self.hid_dim, out_channels=self.hid_dim, kernel_size=3, stride=1, dilation=2, padding=2), bn(self.hid_dim), nn.ReLU())
        self.conv_3 = nn.Sequential(nn.Conv1d(in_channels=self.hid_dim, out_channels=self.hid_dim, kernel_size=3, stride=1, dilation=3, padding=3), bn(self.hid_dim), nn.ReLU())

    def forward(self, x):
        # x: (B, T, F)

        # cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x1 = self.fc_v(x)
        # x1 = torch.cat([cls_tokens, x1], dim=1)  # (B, T + 1, F)
        out_long = self.cma(x1)
        # cls_out = self.mlp_head(out_long[:, 0])
        # out_long = out_long[:, 1:]

        x2 = self.fc_v2(x)
        x2 = x2.permute(0, 2, 1)
        out1 = self.conv_1(x2) + x2
        out2 = self.conv_2(x2) + x2
        out3 = self.conv_3(x2) + x2
        out_short = torch.cat([out1, out2, out3], dim=1)
        out_short = out_short.permute(0, 2, 1)

        return torch.cat([out_long, out_short], dim=2)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        n_features = args.feature_size
        n_class = args.num_classes

        hid_dim = 128
        ffn_dim = hid_dim
        nhead = 4
        dropout = 0.1

        self.fc_v = nn.Linear(1024, hid_dim)
        self.fc_a = nn.Linear(128, hid_dim)

        # self.self_attention_v = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))
        # self.self_attention_a = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))

        # self.temporal_consensus_v = TemporalConsensus(1024, hid_dim)
        # self.temporal_consensus_a = TemporalConsensus(128, hid_dim)
        self.co_attention = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))

        # self.classifier_v = nn.Conv1d(hid_dim * 4, 1, 7, padding=0)
        # self.classifier_a = nn.Conv1d(hid_dim * 4, 1, 7, padding=0)

        # self.fc = nn.Linear(1024 + 128, hid_dim)
        # self.self_attention = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))
        self.temporal_consensus = TemporalConsensus(hid_dim + hid_dim, hid_dim)
        self.classifier = nn.Conv1d(hid_dim * 4, 1, 7, padding=0)
        # self.classifier = nn.Conv1d(hid_dim * 8, 1, 7, padding=0)

        self.apply(weight_init)

    def forward(self, x):
        f = x
        f_v = x[:, :, :1024]  # (b, n, 1024)
        f_a = x[:, :, 1024:]  # (b, n, 128)

        new_v = self.fc_v(f_v)  # (b, n, 128)
        new_a = self.fc_a(f_a)  # (b, n, 128)

        # new_v, new_a = self.co_attention(new_v, new_a)  # (b, n, 128)

        # new_v = self.temporal_consensus_v(new_v)  # (b, n, 128)
        # new_a = self.temporal_consensus_a(new_a)  # (b, n, 128)

        # new_v = new_v.permute(0, 2, 1)  # (b, 128, n)
        # new_a = new_a.permute(0, 2, 1)  # (b, 128, n)

        # logits_v = self.classifier_v(new_v).squeeze(1)  # (b, 1, n) 
        # logits_a = self.classifier_a(new_a).squeeze(1)  # (b, 1, n)

        # logits_v = torch.sigmoid(logits_v)  # (b, n)
        # logits_a = torch.sigmoid(logits_a)  # (b, n)

        # return logits_v, logits_a

        # new_f = self.temporal_consensus(f).permute(0, 2, 1)  # (b, 128, n)
        # new_f = F.pad(new_f, (6, 0))  # padding for conv1d, (b, 128, 6 + n)
        # logits = self.classifier(new_f).squeeze(1)  # (b, 1, n)
        # logits = torch.sigmoid(logits)  # (b, n)

        new_f = torch.cat([new_v, new_a], dim=2)  # (b, n, 128)

        new_f = self.temporal_consensus(new_f).permute(0, 2, 1)  # (b, 128, n)

        new_f = F.pad(new_f, (6, 0))  # padding for conv1d, (b, 128, 6 + n)
        logits = self.classifier(new_f).squeeze(1)  # (b, 1, n)
        logits = torch.sigmoid(logits)  # (b, n)

        return logits, new_v, new_a
