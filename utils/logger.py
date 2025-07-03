import sys
import torch
import time
import numpy as np

from utils.flow_viz import flow_tensor_to_image
from utils.EventToImage import voxel_to_rgb
from utils.disparity_vis import tensor_to_disparity_magma_image

class Logger:
    def __init__(self, lr_scheduler, summary_writer, summary_freq=100, val_freq=1000, start_step=0, metric_frequency=100):
        self.lr_scheduler = lr_scheduler
        self.total_steps = start_step
        self.running_loss = {}
        self.summary_writer = summary_writer
        self.summary_freq = summary_freq
        self.val_freq = val_freq
        self.metric_frequency = metric_frequency

    def print_training_status(self):
        struct = time.localtime()
        print_string = '%d-%02d-%02d %02d:%02d:%02d | step: %06d' % (struct.tm_year, struct.tm_mon, struct.tm_mday, struct.tm_hour, struct.tm_min, struct.tm_sec, self.total_steps)
        for key, value in self.running_loss.items():
            print_string = print_string + '\t %s : %.3f' % (key,  value / self.metric_frequency)
        print(print_string)

    def summary(self, mode='train'):
        # lr
        lr = self.lr_scheduler.get_last_lr()[0]
        self.summary_writer.add_scalar('lr', lr, self.total_steps)
        # loss
        for k in self.running_loss:
            self.summary_writer.add_scalar(mode + '/' + k, self.running_loss[k] / self.metric_frequency, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics, mode='train'):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.summary_freq == 0:
            self.print_training_status()
            self.summary(mode)
            sys.stdout.flush()

    def add_voxel_summary(self, voxel, name):
        if self.total_steps % self.val_freq == 0:
            img1 = torch.from_numpy(voxel_to_rgb(voxel.cpu().detach().numpy())).type(torch.uint8)
            self.summary_writer.add_image(name, img1, self.total_steps)
    
    def add_flow_summary(self, flow, name):
        if self.total_steps % self.val_freq == 0:
            flow_rgb = torch.from_numpy(flow_tensor_to_image(flow))
            self.summary_writer.add_image(name, flow_rgb, self.total_steps)

    def add_disparity_summary(self, disparity, name, vmax=100):
        if self.total_steps % self.val_freq == 0:
            disparity_rgb = torch.from_numpy(np.array(tensor_to_disparity_magma_image(disparity.cpu().detach(), vmax=vmax)).transpose(2,0,1))
            self.summary_writer.add_image(name, disparity_rgb, self.total_steps)

    def write_dict(self, results, tag='validate'):
        for key in results:
            full_tag = tag + '/' + key
            self.summary_writer.add_scalar(full_tag, results[key], self.total_steps)
    
    def close(self):
        self.summary_writer.close()


# import sys
# import torch
# import time

# from utils.flow_viz import flow_tensor_to_image
# from utils.EventToImage import voxel_to_rgb


# class Logger:
#     def __init__(self, lr_scheduler,
#                  summary_writer,
#                  summary_freq=100,
#                  start_step=0,
#                  ):
#         self.lr_scheduler = lr_scheduler
#         self.total_steps = start_step
#         self.running_loss = {}
#         self.summary_writer = summary_writer
#         self.summary_freq = summary_freq

#     def print_training_status(self, mode='train'):
#         struct = time.localtime()
#         print('%d-%02d-%02d %02d:%02d:%02d | step: %06d \t epe: %.3f' % (
#             struct.tm_year, struct.tm_mon, struct.tm_mday, struct.tm_hour, struct.tm_min, struct.tm_sec,
#             self.total_steps, self.running_loss['epe'] / self.summary_freq))

#         for k in self.running_loss:
#             self.summary_writer.add_scalar(mode + '/' + k, self.running_loss[k] / self.summary_freq, self.total_steps)
#             self.running_loss[k] = 0.0

#     def lr_summary(self):
#         lr = self.lr_scheduler.get_last_lr()[0]
#         self.summary_writer.add_scalar('lr', lr, self.total_steps)

#     def add_image_summary(self, voxel_0, voxel_1, flow_preds, flow_gt, mode='train',):
#         if self.total_steps % self.summary_freq == 0:
#             # img_concat = torch.cat((img1[0].detach().cpu(), img2[0].detach().cpu()), dim=-1)
#             # img_concat = img_concat.type(torch.uint8)  # convert to uint8 to visualize in tensorboard

#             img1 = torch.from_numpy(voxel_to_rgb(voxel_0[0].cpu().detach().numpy()))
#             img2 = torch.from_numpy(voxel_to_rgb(voxel_1[0].cpu().detach().numpy()))
#             img_concat = torch.cat((img1, img2), dim=-1)
#             img_concat = img_concat.type(torch.uint8)  # convert to uint8 to visualize in tensorboard
            
#             flow_pred = flow_tensor_to_image(flow_preds[-1][0])
#             forward_flow_gt = flow_tensor_to_image(flow_gt[0])
#             flow_concat = torch.cat((torch.from_numpy(flow_pred),
#                                      torch.from_numpy(forward_flow_gt)), dim=-1)

#             concat = torch.cat((img_concat, flow_concat), dim=-2)

#             self.summary_writer.add_image(mode + '/img_pred_gt', concat, self.total_steps)

#     def push(self, metrics, mode='train'):
#         self.total_steps += 1

#         self.lr_summary()

#         for key in metrics:
#             if key not in self.running_loss:
#                 self.running_loss[key] = 0.0

#             self.running_loss[key] += metrics[key]

#         if self.total_steps % self.summary_freq == 0:
#             self.print_training_status(mode)
#             self.running_loss = {}
#             sys.stdout.flush()

#     def write_dict(self, results):
#         for key in results:
#             tag = key.split('_')[0]
#             tag = tag + '/' + key
#             self.summary_writer.add_scalar(tag, results[key], self.total_steps)

#     def close(self):
#         self.summary_writer.close()
