import torch
from loss import CLAS
from CMA_MIL import CMAL


def train(dataloader, model, optimizer, criterion):
    t_loss = []

    with torch.set_grad_enabled(True):
        model.train()
        for i, (inputs, label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(inputs), dim=2)[0] > 0, 1)
            inputs = inputs[:, :torch.max(seq_len), :]
            inputs = inputs.float().cuda(non_blocking=True)
            label = label.float().cuda(non_blocking=True)

            logits, visual_rep, audio_rep = model(inputs)

            lamda_a2b = 0.05
            lamda_a2n = 0.075
            cmaloss_a2v_a2b, cmaloss_a2v_a2n, cmaloss_v2a_a2b, cmaloss_v2a_a2n = CMAL(label, logits, seq_len, audio_rep, visual_rep)

            clsloss = CLAS(logits, label, seq_len, criterion)

            loss = clsloss + lamda_a2b * cmaloss_a2v_a2b + lamda_a2b * cmaloss_v2a_a2b + lamda_a2n * cmaloss_a2v_a2n + lamda_a2n * cmaloss_v2a_a2n

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss.append(loss)

    print(f'loss {loss:.4f}\tclsloss {clsloss:.4f}\tcmaloss_a2v_a2b {cmaloss_a2v_a2b:.4f}\tcmaloss_v2a_a2b {cmaloss_v2a_a2b:.4f}\tcmaloss_a2v_a2n {cmaloss_a2v_a2n:.4f}\tcmaloss_v2a_a2n {cmaloss_v2a_a2n:.4f}')

    return sum(t_loss)/len(t_loss)


# def train(dataloader, model, optimizer, criterion):
#     t_loss = []

#     with torch.set_grad_enabled(True):
#         model.train()
#         for i, (inputs, label) in enumerate(dataloader):
#             seq_len = torch.sum(torch.max(torch.abs(inputs), dim=2)[0] > 0, 1)
#             inputs = inputs[:, :torch.max(seq_len), :]
#             inputs = inputs.float().cuda(non_blocking=True)
#             label = label.float().cuda(non_blocking=True)

#             logits_v, logits_a = model(inputs)
#             loss_v = CLAS(logits_v, label, seq_len, criterion)
#             loss_a = CLAS(logits_a, label, seq_len, criterion)
#             loss = loss_v + loss_a

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             t_loss.append(loss)

#     print(f'loss_v {loss_v:.4f}\tloss_a {loss_a:.4f}')

#     return sum(t_loss)/len(t_loss)
