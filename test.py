from sklearn.metrics import auc, precision_recall_curve
import numpy as np
import torch


def test(dataloader, model, gt):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()

        for i, inputs in enumerate(dataloader):
            inputs = inputs.cuda()

            logits = model(inputs)
            logits = torch.mean(logits, 0)
            pred = torch.cat((pred, logits))

        pred = list(pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(recall, precision)

        return pr_auc


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

