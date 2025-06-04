import torch
import torch.nn as nn


class SBS(nn.Module):
    def __init__(self, device="cpu"):
        super(SBS, self).__init__()
        self.device = device

    #  直接在原视频里裁出前景运动员,对上面的做出一些改进，背景要使用bboxresize之前的，防止背景的黑色区域太大
    def forward(self, video_clips, bbox):
        # return video_clips : B, C, T, H, W
        B, C, T, H, W = video_clips.shape
        lst = [self.split_video_tensor(video_clips[i], bbox[i]) for i in range(B)]
        mask_before_resize = [row[0] for row in lst]
        mask = [row[1] for row in lst]

        mask = torch.cat(mask)
        mask = mask.unsqueeze(1).cuda()
        mask_before_resize = torch.cat(mask_before_resize)
        mask_before_resize = mask_before_resize.unsqueeze(1).cuda()

        index = torch.randperm(B, device=self.device)
        bg_mask = 1 - (mask_before_resize[index].int() | mask.int())

        video_fuse = video_clips[index] * bg_mask.reshape(-1, 1, 103, H, W) + video_clips * mask.reshape(-1, 1, 103,
                                                                                                         H, W)
        return video_fuse

    # 定义函数来将视频tensor分为前景和后景两部分
    def split_video_tensor(self, video_tensor, bbox_list):
        # 获取视频tensor的形状
        _, num_frames, height, width = video_tensor.shape
        # 将bbox的坐标信息转换为掩码
        fg_mask = torch.zeros((1, num_frames, height, width),
                              dtype=torch.float32)  # shape: (1, num_frames, height, width)
        fg_mask_before_resize = torch.zeros((1, num_frames, height, width), dtype=torch.float32)
        for i in range(num_frames):
            bbox = bbox_list[i]
            x = bbox[[0, 2, 4, 6]]
            y = bbox[[1, 3, 5, 7]]
            x_min = torch.min(x)
            y_min = torch.min(y)
            x_max = torch.max(x)
            y_max = torch.max(y)
            fg_mask_before_resize[0, i, y_min.int().item():y_max.int().item() + 1,
            x_min.int().item():x_max.int().item() + 1] = 1.0

            x_min, y_min, x_max, y_max = self.resize_bbox(x_min, y_min, x_max, y_max)
            x_min = int(max(x_min.item(), 0))
            y_min = int(max(y_min.item(), 0))
            x_max = int(min(x_max.item(), width))
            y_max = int(min(y_max.item(), height))

            fg_mask[0, i, y_min:y_max + 1, x_min:x_max + 1] = 1.0
        return [fg_mask_before_resize, fg_mask]

    def resize_bbox(self, x1, y1, x2, y2, ratio=1):
        # 计算原始bbox的中心点坐标
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 计算原始bbox的宽度和高度
        width = x2 - x1
        height = y2 - y1

        # 将宽度和高度分别放大ratio倍
        new_width = width * ratio
        new_height = height * ratio

        # 计算新的bbox左上角和右下角的坐标
        new_x1 = center_x - new_width / 2
        new_y1 = center_y - new_height / 2
        new_x2 = center_x + new_width / 2
        new_y2 = center_y + new_height / 2

        return new_x1, new_y1, new_x2, new_y2


