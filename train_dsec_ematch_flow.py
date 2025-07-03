import argparse
import os
import torch
import yaml
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from datasets.DSEC.DSECProvider import DSECProvider

from models.ematch.ematch import EventMatch

from loss.loss_flow import flow_loss_func

from utils import misc
from utils.logger import Logger


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--checkpoint_dir', type=str, help='where to save the training log and models')

    # configs
    parser.add_argument('--configs_model', default='./models/configs_model/ematch/flow.yaml', type=str)
    parser.add_argument('--configs_dataset', default='./datasets/configs_dataset/ematch/flow/dsec_train.yaml', type=str)

    # dataloader
    parser.add_argument('--dataloader_num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=4, type=int)

    # training
    parser.add_argument('--num_steps', default=400000, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--pct_start', default=0.01, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)

    # loss
    parser.add_argument('--gamma', default=0.8, type=float, help='exponential weighting')
    parser.add_argument('--max_flow', default=400, type=int, help='exclude very large motions during training')

    # validate
    parser.add_argument('--summary_freq', default=1000, type=int)
    parser.add_argument('--save_ckpt_freq', default=10000, type=int)
    parser.add_argument('--val_freq', default=10000, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str, help='resume from pretrain model for finetuing or resume from terminated training')
    parser.add_argument('--strict_resume', action='store_true')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    return parser

def main(args):
    # Inital setting
    misc.save_args(args)       # save arguments to 'args.json'
    misc.save_command(args.checkpoint_dir)   # save commands to 'command_train.txt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Use %d GPUs' % torch.cuda.device_count())
    print('pytorch version:', torch.__version__)
    print(args)
    
    # Seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    # ***************************************************************************** 1.Model *****************************************************************************
    with open(args.configs_model, "r") as f: model_config = yaml.safe_load(f)
    model = EventMatch(model_config).to(device)
    model_without_dp = model
    model.to(device)
    print('Model definition:')
    print(model)

    # ***************************************************************************** 2.Optimizer *****************************************************************************
    param_dicts = model_without_dp.parameters()
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr)
    num_params = sum(p.numel() for p in model.parameters())
    open(os.path.join(args.checkpoint_dir, '%d_parameters' % num_params), 'a').close()
    print('Number of params:', num_params)

    # Resume checkpoints
    start_epoch = 0
    start_step = 0
    if args.resume:
        print('Load checkpoint: %s' % args.resume)

        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

        model_without_dp.load_state_dict(weights, strict=args.strict_resume)
        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not args.no_resume_optimizer:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']

        print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))

    # ***************************************************************************** 3.Dataloader *****************************************************************************
    # DataLoader
    with open(args.configs_dataset, "r") as f: config_dataset = yaml.safe_load(f)
    dataset = DSECProvider(config_dataset).get_datasets()
    train_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.dataloader_num_workers)
    print('Number of training images:', len(train_loader))

    # ***************************************************************************** 4.Scheduler *****************************************************************************
    # Scheduler    
    last_epoch = start_step if args.resume and start_step > 0 else -1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = args.lr,
        total_steps = args.num_steps + 10000,
        pct_start=args.pct_start, 
        cycle_momentum=False, 
        anneal_strategy='cos',
        last_epoch=last_epoch,
    )

    # Logger
    summary_writer = SummaryWriter(args.checkpoint_dir)
    logger = Logger(lr_scheduler, summary_writer, args.summary_freq, args.val_freq, start_step=start_step, metric_frequency=args.summary_freq)

    # Training
    epoch = start_epoch
    total_steps = start_step
    print('Start training')
    while total_steps < args.num_steps:
        model.train()
        for i, sample in enumerate(train_loader):
            # ***************************************************************************** 5.Data *****************************************************************************
            voxel_0 = sample['voxel_0'].to(device)
            voxel_1 = sample['voxel_1'].to(device)
            flow_gt = sample['target'].to(device)
            valid = sample['valid'].to(device)

            # ************************************************************************ 6. Input/Output *************************************************************************
            results_dict = model(voxel_0, voxel_1)
            flow_preds = results_dict['flow_preds']

            # *************************************************************************** 7. Loss ******************************************************************************
            # loss
            metrics_log = {}
            loss, metrics = flow_loss_func(flow_preds, flow_gt, valid, gamma=args.gamma, max_flow=args.max_flow) # Loss1

            if torch.isnan(loss):
                continue

            metrics_log.update({'flow_loss': loss.item()})
            metrics_log.update(metrics)

            # more efficient zero_grad
            for param in model_without_dp.parameters():
                param.grad = None

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            lr_scheduler.step()

            # *************************************************************************** 8. Logger ******************************************************************************
            logger.push(metrics_log)
            logger.add_voxel_summary(voxel_0[0], 'train/voxel')
            logger.add_flow_summary(flow_preds[-1][0], 'train/flow_predicted')
            logger.add_flow_summary(flow_gt[0], 'train/flow_gt')

            total_steps += 1
            if total_steps % args.save_ckpt_freq == 0 or total_steps == args.num_steps:
                checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)
                torch.save(model_without_dp.state_dict(), checkpoint_path)

            if total_steps % args.save_latest_ckpt_freq == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')
                torch.save({
                    'model': model_without_dp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': total_steps,
                    'epoch': epoch,
                }, checkpoint_path)

            # # Validation
            # if total_steps % cfgs.val_freq == 0:
            #     print('Start validation')

            if total_steps >= args.num_steps:
                print('Training done')
                return

        epoch += 1

    checkpoint_path = os.path.join(args.checkpoint_dir, 'final.pth')
    torch.save(model_without_dp.state_dict(), checkpoint_path)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
