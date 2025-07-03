import argparse
import torch
import numpy as np
import os
import cv2
import yaml
from tqdm import tqdm

from datasets.MVSEC.MVSECProvider import MVSECProvider

from models.ematch.ematch import EventMatch

from utils.metric import Metric
from utils.flow_viz import flow_tensor_to_image
from utils.disparity_vis import tensor_to_disparity_magma_image, tensor_to_disparity_jet_image
from utils.errors import compute_n_pixels_error, compute_absolute_error
from utils.EventToImage import voxel_to_rgb


def get_args_parser():
    parser = argparse.ArgumentParser()
    # initial setting
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save_dir', default='tmp', type=str, help='where to save outcomes of evaluation')
    parser.add_argument('--visual', default=True, type=bool)
    parser.add_argument('--real', default=False, action='store_true')
    parser.add_argument('--event_voxel', default=False, action='store_true')
    
    # configs
    parser.add_argument('--configs_model', default='./models/configs_model/ematch/disparity.yaml', type=str)
    parser.add_argument('--configs_dataset', default='./datasets/configs_dataset/ematch/disparity/mvsec_test_split1.yaml', type=str)

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
    print('Use %d GPUs' % torch.cuda.device_count())
    model_without_dp = model

    # load weights
    checkpoint = torch.load(cfgs.checkpoint, map_location=device)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model_without_dp.load_state_dict(weights, strict=False)

    # testing
    with open(args.configs_dataset, "r") as f: config_dataset = yaml.safe_load(f)
    test_datasets, names = MVSECProvider(config_dataset).get_datasets(True)
    for i, name in enumerate(names):
        test_dataset = test_datasets[i]
        
        # save_path
        visual_path = os.path.join(cfgs.save_dir, 'visual', name)
        metrics_path = os.path.join(cfgs.save_dir, 'metrics')
        if not os.path.exists(visual_path): os.makedirs(visual_path)
        if not os.path.exists(metrics_path): os.makedirs(metrics_path)

        if cfgs.event_voxel:
            voxel_path = os.path.join(cfgs.save_dir, 'voxel', name)
            if not os.path.exists(voxel_path): os.makedirs(voxel_path)
        if cfgs.real:
            real_path = os.path.join(cfgs.save_dir, 'real', name)
            if not os.path.exists(real_path): os.makedirs(real_path)

        # Load Metric
        # metric = Metric('torch')
        one_pixel_error_list = []
        mean_disparity_error_list = []
        mean_depth_error_list = []
        median_depth_error_list = []

        # Outdoor = True if name == 'outdoor_day1' or name == 'outdoor_day2' else False
        # frame_rate = 45 if name == 'outdoor_day1' or name == 'outdoor_day2' else 30
        # video_frames_flowp = []
        # video_frames_flowr = []

        # model
        model = model.eval()
        for i in tqdm(range(len(test_dataset))):
            sample = test_dataset[i]
            voxel_0 = sample['voxel_0'][None].to(device)
            voxel_1 = sample['voxel_1'][None].to(device)
            disparity_real = sample['target'][None].to(device)
            valid = sample['valid'][None].to(device)

            ####################################################################################################################################
            if cfgs.real:
                ground_truth_disparity = disparity_real[0]
                disparity_pred = tensor_to_disparity_jet_image(ground_truth_disparity.cpu().detach(), vmax=37)
                disparity_pred.save(os.path.join(real_path, str(i).zfill(6) + '.png'))
                # continue
            ####################################################################################################################################
            if cfgs.event_voxel:
                voxel_rgb_0 = voxel_to_rgb(voxel_0[0].cpu().detach().numpy()).transpose(1,2,0)
                cv2.imwrite(os.path.join(voxel_path,  str(i).zfill(6) + '_left.png'), cv2.cvtColor(voxel_rgb_0, cv2.COLOR_RGB2BGR))
                voxel_rgb_1 = voxel_to_rgb(voxel_1[0].cpu().detach().numpy()).transpose(1,2,0)
                cv2.imwrite(os.path.join(voxel_path,  str(i).zfill(6) + '_right.png'), cv2.cvtColor(voxel_rgb_1, cv2.COLOR_RGB2BGR))
                # continue
            ####################################################################################################################################
            
            results_dict = model(voxel_0, voxel_1, task='disparity')
            disparity_preds = results_dict['disparity_preds'][-1][0]
            
            if cfgs.visual:
                disparity_pred = tensor_to_disparity_jet_image(disparity_preds.cpu().detach(), vmax=37)
                disparity_pred.save(os.path.join(visual_path, str(i).zfill(6) + '.png'))

            estimated_disparity = disparity_preds
            ground_truth_disparity = disparity_real[0]
        
            original_dataset = test_dataset.mvsec_left
            estimated_depth = original_dataset.disparity_to_depth(estimated_disparity)
            ground_truth_depth = original_dataset.disparity_to_depth(ground_truth_disparity)
            binary_error_map, one_pixel_error = compute_n_pixels_error(estimated_disparity, ground_truth_disparity, n=1.0)
            mean_disparity_error = compute_absolute_error(estimated_disparity, ground_truth_disparity)[1]
            mean_depth_error = compute_absolute_error(estimated_depth, ground_truth_depth)[1]
            median_depth_error = compute_absolute_error(estimated_depth,ground_truth_depth, use_mean=False)[1]
            
            one_pixel_error_list.append(one_pixel_error)
            mean_disparity_error_list.append(mean_disparity_error)
            mean_depth_error_list.append(mean_depth_error)
            median_depth_error_list.append(median_depth_error)

        with open(os.path.join(metrics_path, name+".txt"), 'w') as f:
            one_pixel_error = np.mean(one_pixel_error_list)
            mean_disparity_error = np.mean(mean_disparity_error_list)
            mean_depth_error = np.mean(mean_depth_error_list)
            median_depth_error = np.mean(median_depth_error_list)
            f.write('one_pixel_error=' + str(one_pixel_error) + '\n')
            f.write('mean_disparity_error=' + str(mean_disparity_error) + '\n')
            f.write('mean_depth_error=' + str(mean_depth_error) + '\n')
            f.write('median_depth_error=' + str(median_depth_error) + '\n')
            print('one_pixel_error=' + str(one_pixel_error))
            print('mean_disparity_error=' + str(mean_disparity_error))
            print('mean_depth_error=' + str(mean_depth_error))
            print('median_depth_error=' + str(median_depth_error))

        # if True:
        #     video_size = (256, 256)
        #     # 2. 设置视频写入器
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
        #     # 完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
        #     videowrite = cv2.VideoWriter(os.path.join(cfgs.save_path, "Flow_real.mp4"), fourcc, frame_rate, video_size)  # 2是每秒的帧数，size是图片尺寸

        #     # 3.迭代处理图像
        #     for i in tqdm(range(len(video_frames_flowr))):
        #         videowrite.write(video_frames_flowr[i])  # 写入视频
        #     videowrite.release()

        #     # 2. 设置视频写入器
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
        #     # 完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
        #     videowrite = cv2.VideoWriter(os.path.join(cfgs.save_path, "Flow_pre.mp4"), fourcc, frame_rate, video_size)  # 2是每秒的帧数，size是图片尺寸

        #     # 3.迭代处理图像
        #     for i in tqdm(range(len(video_frames_flowp))):
        #         videowrite.write(video_frames_flowp[i])  # 写入视频
        #     videowrite.release()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    with torch.no_grad():
        main(args)
