from sklearn.metrics import auc, precision_recall_curve
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


matplotlib.rc("font", family='Times New Roman')  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示符号
matplotlib.rcParams['font.size'] = 10.0

chinese_font_properties = FontProperties(fname='C:/Windows/Fonts/simsun.ttc')


def test(dataloader, model, gt):
    all_raw_visual_features = []
    all_raw_audio_features = []
    all_visual_features = []
    all_audio_features = []
    all_labels = []

    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()

        start_index = 0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.cuda()

            logits, visual_rep, audio_rep = model(inputs)
            logits = torch.mean(logits, 0)
            pred = torch.cat((pred, logits))

            all_raw_visual_features.append(inputs[:, :, :1024].mean(0).mean(0).cpu())
            all_raw_audio_features.append(inputs[:, :, 1024:].mean(0).mean(0).cpu())
            all_visual_features.append(visual_rep.mean(0).mean(0).cpu())
            all_audio_features.append(audio_rep.mean(0).mean(0).cpu())
            all_labels.append(labels.mean(0, keepdim=True).cpu())

            y = np.repeat(logits.cpu().numpy(), 16)
            y_gt = gt[start_index:start_index + logits.shape[0] * 16]
            x = np.arange(logits.shape[0] * 16)
            y_1 = np.ones_like(x)
            y_0 = np.zeros_like(x)

            plt.figure(1, figsize=(8, 4))
            plt.clf()
            plt.plot(x, y)
            plt.ylim(0, 1)
            plt.xlabel("时间（帧）", fontproperties=chinese_font_properties)
            plt.ylabel("异常分数", fontproperties=chinese_font_properties)
            plt.fill_between(x, y_1, y_0, where=(y_gt == 1), color='red', alpha=0.1)
            plt.tight_layout()
            plt.show()
            start_index += logits.shape[0] * 16

        pred = list(pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(recall, precision)

        all_raw_visual_features = torch.stack(all_raw_visual_features)
        all_raw_audio_features = torch.stack(all_raw_audio_features)
        all_visual_features = torch.stack(all_visual_features)
        all_audio_features = torch.stack(all_audio_features)
        all_labels = torch.cat(all_labels)

        ret = (all_raw_visual_features, all_raw_audio_features, all_visual_features, all_audio_features, all_labels)

        return pr_auc, ret


# def test(dataloader, model, gt):
#     with torch.no_grad():
#         model.eval()
#         pred = torch.zeros(0).cuda()
#         pred_v = torch.zeros(0).cuda()
#         pred_a = torch.zeros(0).cuda()

#         for i, inputs in enumerate(dataloader):
#             inputs = inputs.cuda()

#             logits_v, logits_a = model(inputs)
#             logits_v = torch.mean(logits_v, 0)
#             logits_a = torch.mean(logits_a, 0)
#             pred_v = torch.cat((pred_v, logits_v))
#             pred_a = torch.cat((pred_a, logits_a))
#             pred = torch.cat((pred, (logits_v + logits_a) / 2))

#         pred_v = list(pred_v.cpu().detach().numpy())
#         pred_a = list(pred_a.cpu().detach().numpy())
#         pred = list(pred.cpu().detach().numpy())
#         precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred, 16))
#         pr_auc = auc(recall, precision)
#         precision_v, recall_v, th_v = precision_recall_curve(list(gt), np.repeat(pred_v, 16))
#         pr_auc_v = auc(recall_v, precision_v)
#         precision_a, recall_a, th_a = precision_recall_curve(list(gt), np.repeat(pred_a, 16))
#         pr_auc_a = auc(recall_a, precision_a)

#         print(f'ap {pr_auc:.4f}\tap_v {pr_auc_v:.4f}\tap_a {pr_auc_a:.4f}')

#         return pr_auc

