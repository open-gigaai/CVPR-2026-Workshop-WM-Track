import torch
from loss import VideoDepthLoss

B = 2
T = 32
H = 518
W = 518

prediction = torch.randn(B, T, H, W).cuda()
target = torch.randn(B, T, H, W).cuda()
mask = torch.ones(B, T, H, W).bool().cuda()

loss = VideoDepthLoss()
val = loss(prediction, target, mask)
print(f'loss: {val}')
