import numpy as np
import torch
import torch.nn.functional as F


def disparity_loss_func(pred_disps, gt_disp, mask, 
                        gamma=0.9, 
                        max_disp=400,
                        **kwargs):
    n_predictions = len(pred_disps)
    total_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mask = (mask >= 0.5) & (gt_disp < max_disp)

    # loss weights
    loss_weights = [gamma ** (len(pred_disps) - 1 - power) for power in range(n_predictions)]

    for k in range(n_predictions):
        pred_disp = pred_disps[k]
        weight = loss_weights[k]

        curr_loss = F.smooth_l1_loss(pred_disp[mask], gt_disp[mask], reduction='mean')
        total_loss += weight * curr_loss

    epe = torch.abs(pred_disps[-1] - gt_disp)

    epe = epe.view(-1)[mask.view(-1)]

    metrics = {
        'disparity_epe': epe.mean().item(),
        'disparity_1px': (epe > 1).float().mean().item(),
        'disparity_3px': (epe > 3).float().mean().item(),
        'disparity_5px': (epe > 5).float().mean().item(),
    }

    return total_loss, metrics
