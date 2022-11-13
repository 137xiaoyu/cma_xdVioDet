from torch.utils.data import DataLoader
import torch
import numpy as np
from model import Model
from dataset import Dataset
from test import test
import option
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':
    print('perform testing...')
    args = option.parser.parse_args()
    device = torch.device("cuda")

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=5, shuffle=False,
                              num_workers=0, pin_memory=True)
    model = Model(args)
    model = model.to(device)

    if args.ckpt:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint)

    gt = np.load(args.gt)
    st = time.time()

    pr_auc, ret = test(test_loader, model, gt)
    time_elapsed = time.time() - st
    print('test AP: {:.4f}\n'.format(pr_auc))
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    raw_visual_features, raw_audio_features, visual_features, audio_features, labels = ret

    feature_save_dir = './features_81.607/'
    if not os.path.exists(feature_save_dir):
        os.makedirs(feature_save_dir)

    # np.savetxt(os.path.join(feature_save_dir, 'raw_visual_features.txt'), raw_visual_features)
    # np.savetxt(os.path.join(feature_save_dir, 'raw_audio_features.txt'), raw_audio_features)
    np.savetxt(os.path.join(feature_save_dir, 'visual_features.txt'), visual_features)
    np.savetxt(os.path.join(feature_save_dir, 'audio_features.txt'), audio_features)
    np.savetxt(os.path.join(feature_save_dir, 'labels.txt'), labels)
