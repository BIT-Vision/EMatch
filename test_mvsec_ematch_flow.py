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
    parser.add_argument('--configs_model', default='./models/configs_model/ematch/flow.yaml', type=str)
    parser.add_argument('--configs_dataset', default='./datasets/configs_dataset/ematch/flow/mvsec_test.yaml', type=str)

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
        metric = Metric('torch')
        epe_list = []
        _1pe_list = []
        _2pe_list = []
        _3pe_list = []
        outlier_list = []

        epe_list_sparse = []
        _1pe_list_sparse = []
        _2pe_list_sparse = []
        _3pe_list_sparse = []
        outlier_list_sparse = []
        Outdoor = True if name == 'outdoor_day1' or name == 'outdoor_day2' else False
        frame_rate = 45 if name == 'outdoor_day1' or name == 'outdoor_day2' else 30
        video_frames_flowp = []
        video_frames_flowr = []

        # model
        model = model.eval()
        for i in tqdm(range(len(test_dataset))):
            sample = test_dataset[i]
            voxel_0 = sample['voxel_0'][None].to(device)
            voxel_1 = sample['voxel_1'][None].to(device)
            flow_real = sample['target'][None].to(device)
            valid = sample['valid'][None].to(device)

            ####################################################################################################################################
            if cfgs.real:
                ground_truth_flow = flow_tensor_to_image(flow_real[0]).transpose(1,2,0)
                cv2.imwrite(os.path.join(real_path, str(i).zfill(6) + '.png'), cv2.cvtColor(ground_truth_flow, cv2.COLOR_RGB2BGR))
                # continue
            ####################################################################################################################################
            if cfgs.event_voxel:
                voxel_rgb_0 = voxel_to_rgb(voxel_0[0].cpu().detach().numpy()).transpose(1,2,0)
                cv2.imwrite(os.path.join(voxel_path,  str(i).zfill(6) + '_left.png'), cv2.cvtColor(voxel_rgb_0, cv2.COLOR_RGB2BGR))
                voxel_rgb_1 = voxel_to_rgb(voxel_1[0].cpu().detach().numpy()).transpose(1,2,0)
                cv2.imwrite(os.path.join(voxel_path,  str(i).zfill(6) + '_right.png'), cv2.cvtColor(voxel_rgb_1, cv2.COLOR_RGB2BGR))
                # continue
            ####################################################################################################################################
            
            results_dict = model(voxel_0, voxel_1)
            flow_predicted = results_dict['flow_preds'][-1]
            
            if cfgs.visual:
                flow_pred_rgb = flow_tensor_to_image(flow_predicted[0]).transpose(1,2,0)
                cv2.imwrite(os.path.join(visual_path, str(i).zfill(6) + '.png'), cv2.cvtColor(flow_pred_rgb, cv2.COLOR_RGB2BGR))

            voxel = voxel_0 + voxel_1
            if Outdoor:
                flow_real = flow_real[:, :, 0:190, :]
                flow_predicted = flow_predicted[:, :, 0:190, :]
                valid = valid[:, 0:190, :]
                voxel = voxel[:, :, 0:190, :]
            
            # EPE
            # Get Flow Mask
            mag = metric.Magnitude(flow_real - flow_predicted)
            mask = (valid >= 0.5) & (mag < 400)
            event_mask = torch.norm(voxel, p=2, dim=1, keepdim=False) > 0
            sparse_mask = torch.logical_and(mask, event_mask)

            # Dense
            epe = metric.EPE(flow_real, flow_predicted, mask)
            aee = metric.AEE(epe)
            epe_list.append(aee.cpu().numpy())

            _1pe = metric.NPE(epe, 1)
            _1pe_list.append(_1pe.cpu().numpy())

            _2pe = metric.NPE(epe, 2)
            _2pe_list.append(_2pe.cpu().numpy())

            _3pe = metric.NPE(epe, 3)
            _3pe_list.append(_3pe.cpu().numpy())

            outlier = metric.Outlier(epe, metric.Magnitude(flow_real, mask))
            outlier_list.append(outlier.cpu().numpy())

            # sparse
            epe = metric.EPE(flow_real, flow_predicted, sparse_mask)
            aee = metric.AEE(epe)
            epe_list_sparse.append(aee.cpu().numpy())

            _1pe = metric.NPE(epe, 1)
            _1pe_list_sparse.append(_1pe.cpu().numpy())

            _2pe = metric.NPE(epe, 2)
            _2pe_list_sparse.append(_2pe.cpu().numpy())

            _3pe = metric.NPE(epe, 3)
            _3pe_list_sparse.append(_3pe.cpu().numpy())

            outlier = metric.Outlier(epe, metric.Magnitude(flow_real, sparse_mask))
            outlier_list_sparse.append(outlier.cpu().numpy())

        with open(os.path.join(metrics_path, name+".txt"), 'w') as f:
            EPE = np.mean(epe_list)
            _1PE = np.mean(_1pe_list)
            _2PE = np.mean(_2pe_list)
            _3PE = np.mean(_3pe_list)
            Outlier = np.mean(outlier_list)
            f.write('EPE=' + str(EPE) + '\n')
            f.write('1pe=' + str(_1PE) + '\n')
            f.write('2pe=' + str(_2PE) + '\n')
            f.write('3pe=' + str(_3PE) + '\n')
            f.write('Outlier=' + str(Outlier) + '\n')
            print('EPE=' + str(EPE))
            print('1pe=' + str(_1PE))
            print('2pe=' + str(_2PE))
            print('3pe=' + str(_3PE))
            print('Outlier=' + str(Outlier))
            EPE = np.mean(epe_list_sparse)
            _1PE = np.mean(_1pe_list_sparse)
            _2PE = np.mean(_2pe_list_sparse)
            _3PE = np.mean(_3pe_list_sparse)
            Outlier = np.mean(outlier_list)
            f.write('EPE_sparse=' + str(EPE) + '\n')
            f.write('1pe_sparse=' + str(_1PE) + '\n')
            f.write('2pe_sparse=' + str(_2PE) + '\n')
            f.write('3pe_sparse=' + str(_3PE) + '\n')
            f.write('Outlier_sparse=' + str(Outlier) + '\n')
            print('EPE_sparse=' + str(EPE))
            print('1pe_sparse=' + str(_1PE))
            print('2pe_sparse=' + str(_2PE))
            print('3pe_sparse=' + str(_3PE))
            print('Outlier_sparse=' + str(Outlier))

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
