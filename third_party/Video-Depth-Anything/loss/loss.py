import torch
import torch.nn as nn
import numpy as np

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return torch.sum(image_loss) * 0.0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based, frame_id_mask=None):
    # mask for distinguish different frames
    valid_id_mask_x = torch.ones_like(mask[:, :, 1:])
    valid_id_mask_y = torch.ones_like(mask[:, 1:, :])
    if frame_id_mask is not None:
        valid_id_mask_x = ((frame_id_mask[:, :, 1:] - frame_id_mask[:, :, :-1]) == 0).to(mask.dtype)
        valid_id_mask_y = ((frame_id_mask[:, 1:, :] - frame_id_mask[:, :-1, :]) == 0).to(mask.dtype)
    
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(torch.mul(mask[:, :, 1:], mask[:, :, :-1]), valid_id_mask_x)
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(torch.mul(mask[:, 1:, :], mask[:, :-1, :]), valid_id_mask_y)
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)

def normalize_prediction_robust(target, mask, ms=None):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    if ms is None:
        m = torch.zeros_like(ssum)
        s = torch.ones_like(ssum)

        m[valid] = torch.median((mask[valid] * target[valid]).view(valid.sum(), -1), dim=1).values
    else:
        m, s = ms

    target = target - m.view(-1, 1, 1)

    if ms is None:
        sq = torch.sum(mask * target.abs(), (1, 2))
        s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

    return target / (s.view(-1, 1, 1)), (m.detach(), s.detach())


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid]
                  * b_1[valid]) / (det[valid] + 1e-6)
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid]
                  * b_1[valid]) / (det[valid] + 1e-6)

    return x_0, x_1

class TrimmedProcrustesLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, trim=0.2, reduction="batch-based"):
        super().__init__()

        self.__data_loss = TrimmedMAELoss(reduction=reduction, trim=trim)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None
        self.__prediction_median_scale = None
        self.__target_median_scale = None

    def forward(self, prediction, target, mask, pred_ms=None, tar_ms=None, num_frame_h=1, no_norm=False):
        if no_norm:
            self.__prediction_ssi, self.__prediction_median_scale = prediction, (0, 1)
            target_, self.__target_median_scale = target, (0, 1)
        else:
            self.__prediction_ssi, self.__prediction_median_scale = normalize_prediction_robust(prediction, mask, ms=pred_ms)
            target_, self.__target_median_scale = normalize_prediction_robust(target, mask, ms=tar_ms)

        total = self.__data_loss(self.__prediction_ssi, target_, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(
                self.__prediction_ssi, target_, mask, num_frame_h=num_frame_h
            )

        return total

    def get_median_scale(self):
        return self.__prediction_median_scale, self.__target_median_scale

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class TrimmedMAELoss(nn.Module):
    def __init__(self, trim=0.2, reduction="batch-based"):
        super().__init__()

        self.trim = trim

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask, weight_mask=None):
        if torch.sum(mask) == 0:
            return torch.sum(prediction) * 0.0
        M = torch.sum(mask, (1, 2))
        res = prediction - target
        if weight_mask is not None:
            res = res * weight_mask
        res = res[mask.bool()].abs()
        trimmed, _ = torch.sort(res.view(-1), descending=False)
        keep_num = int(len(res) * (1.0 - self.trim))
        if keep_num <= 0:
            return torch.sum(prediction) * 0.0
        trimmed = trimmed[: keep_num]

        return self.__reduction(trimmed, M)

    
class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask, num_frame_h=1):
        total = 0

        frame_id_mask = None
        if num_frame_h > 1:
            frame_h = mask.shape[1] // num_frame_h
            frame_id_mask = torch.zeros_like(mask)
            for i in range(num_frame_h):
                frame_id_mask[:, i*frame_h:(i+1)*frame_h, :] = i+1

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self.__reduction,
                frame_id_mask=frame_id_mask[:, ::step, ::step] if num_frame_h > 1 else None,
            )

        return total


class TemporalGradientMatchingLoss(nn.Module):
    def __init__(self, trim=0.2, temp_grad_scales=4, temp_grad_decay=0.5, reduction="batch-based", diff_depth_th=0.05):
        super().__init__()

        self.data_loss = TrimmedMAELoss(trim=trim, reduction=reduction)
        self.temp_grad_scales = temp_grad_scales
        self.temp_grad_decay = temp_grad_decay
        self.diff_depth_th = diff_depth_th

    def forward(self, prediction, target, mask):
        '''
            prediction: Shape(B, T, H, W)
            target: Shape(B, T, H, W)
            mask: Shape(B, T, H, W)
        '''
        total = 0
        cnt = 0

        min_target = torch.where(mask.bool(), target, torch.inf).min(-1).values.min(-1).values
        max_target = torch.where(mask.bool(), target, -torch.inf).max(-1).values.max(-1).values
        target_th = (max_target - min_target) * self.diff_depth_th

        for scale in range(self.temp_grad_scales):
            temp_stride = pow(2, scale)
            if temp_stride < prediction.shape[1]:
                pred_temp_grad = torch.diff(prediction[:,::temp_stride,...], dim=1)
                target_temp_grad = torch.diff(target[:,::temp_stride,...], dim=1)
                temp_mask = mask[:,::temp_stride,...][:,1:,...] & mask[:,::temp_stride,...][:,:-1,...]
                
                valid_mask_from_target_th = target_temp_grad.abs() < target_th.unsqueeze(-1).unsqueeze(-1)[:,::temp_stride,...][:,1:,...]
                temp_mask = temp_mask & valid_mask_from_target_th

                total += self.data_loss(prediction=pred_temp_grad.flatten(0, 1), target=target_temp_grad.flatten(0, 1), mask=temp_mask.flatten(0, 1)) * pow(self.temp_grad_decay, scale)
                cnt += 1

        return total / cnt


class VideoDepthLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, trim=0.0, stable_scale=10, reduction="batch-based"):
        super().__init__()
        self.spatial_loss = TrimmedProcrustesLoss(alpha=alpha, scales=scales, trim=trim, reduction=reduction)
        self.stable_loss = TemporalGradientMatchingLoss(trim=trim, reduction=reduction, temp_grad_decay=0.5, temp_grad_scales=1)
        self.stable_scale = stable_scale

    def forward(self, prediction, target, mask):
        '''
            prediction: Shape(B, T, H, W)
            target: Shape(B, T, H, W)
            mask: Shape(B, T, H, W)
        '''
        loss_dict = {}
        total = 0
        loss_dict['spatial_loss'] = self.spatial_loss(prediction=prediction.flatten(0, 1), target=target.flatten(0, 1), mask=mask.flatten(0, 1).float())
        total += loss_dict['spatial_loss']
        scale, shift = compute_scale_and_shift(prediction.flatten(1,2), target.flatten(1,2), mask.flatten(1,2))
        prediction = scale.view(-1, 1, 1, 1) * prediction + shift.view(-1, 1, 1, 1)
        loss_dict['stable_loss'] = self.stable_loss(prediction=prediction, target=target, mask=mask) * self.stable_scale
        total += loss_dict['stable_loss']

        loss_dict['total_loss'] = total
        return loss_dict
