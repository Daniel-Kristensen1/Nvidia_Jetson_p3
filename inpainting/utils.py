import torch
from PIL import Image

def tensor_to_img(t: torch.Tensor):
    """
    t: (C, H, W) med værdier [0,1]
    return: (H, W, C) numpy-array klar til plt.imshow
    """
    t = t.detach().cpu().clamp(0, 1)
    return t.permute(1, 2, 0).numpy()
