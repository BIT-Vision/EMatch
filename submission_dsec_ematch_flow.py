import argparse
import torch
import numpy as np
import os
import cv2
import yaml
import imageio
from tqdm import tqdm

from datasets.DSEC.DSECProvider import DSECProvider

from models.ematch.ematch import EventMatch

from utils.metric import Metric
from utils.flow_viz import flow_tensor_to_image


def get_args_parser():
    parser = argparse.ArgumentParser()
    # initial setting
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save_dir', default='tmp', type=str, help='where to save outcomes of evaluation')
    parser.add_argument('--visual', default=True, type=bool)

    # configs
    parser.add_argument('--configs_model', default='./models/configs_model/ematch/flow.yaml', type=str)
    parser.add_argument('--configs_dataset', default='./datasets/configs_dataset/ematch/flow/dsec_test.yaml', type=str)

    # checkpoint
    parser.add_argument('--checkpoint', type=str)

    return parser


def main(cfgs):
    # initial setting
    seed = cfgs.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(cfgs.seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    with open(args.configs_model, "r") as f: model_config = yaml.safe_load(f)
    model = EventMatch(model_config).to(device)
    model_without_dp = model

    # load weights
    checkpoint = torch.load(cfgs.checkpoint, map_location=device)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model_without_dp.load_state_dict(weights, strict=False)

    # testing
    with open(args.configs_dataset, "r") as f: config_dataset = yaml.safe_load(f)
    test_datasets, names = DSECProvider(config_dataset).get_datasets(True)
    for i, name in enumerate(names):
        test_dataset = test_datasets[i]

        # save_path
        submission_path = os.path.join(cfgs.save_dir, 'submission', name)
        visual_path = os.path.join(cfgs.save_dir, 'visual', name)
        if not os.path.exists(submission_path): os.makedirs(submission_path)
        if not os.path.exists(visual_path): os.makedirs(visual_path)

        # model
        model = model.eval()
        for i in tqdm(range(len(test_dataset))):
            sample = test_dataset[i]
            voxel_0 = sample['voxel_0'][None].to(device)
            voxel_1 = sample['voxel_1'][None].to(device)
            file_index = sample['file_index']

            results_dict = model(voxel_0, voxel_1)
            flow_preds = results_dict['flow_preds'][-1][0]
            
            if cfgs.visual:
                flow_pred = flow_tensor_to_image(flow_preds).transpose(1,2,0)
                cv2.imwrite(os.path.join(visual_path, str(file_index).zfill(6) + '.png'), cv2.cvtColor(flow_pred, cv2.COLOR_RGB2BGR))

            # Save
            flow_pred = flow_preds.permute(1, 2, 0).detach().cpu().numpy()
            I = np.zeros((480, 640, 3))
            I[...,0] = (flow_pred[...,0] * 128 + 2**15)
            I[...,1] = (flow_pred[...,1] * 128 + 2**15)
            I[...,2] = 0
            I = I.astype(np.uint16)

            imageio.imwrite(os.path.join(submission_path, str(file_index).zfill(6) + '.png'), I, format='PNG-FI')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    with torch.no_grad():
        main(args)
