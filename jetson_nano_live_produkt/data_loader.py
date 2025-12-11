import numpy as np
import torch
from PIL import Image


def _pil_resize(image: np.ndarray, new_h: int, new_w: int, is_label: bool = False) -> np.ndarray:
    """
    Resize et numpy-billede med PIL.
    image: (H, W, C) eller (H, W)
    """
    # Sørg for uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Hvis label er (H, W, 1), så fjern sidste axis
    if is_label and image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    pil_img = Image.fromarray(image)

    if is_label:
        # For mask/label vil vi ikke blende værdierne
        pil_img = pil_img.resize((new_w, new_h), Image.NEAREST)
    else:
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

    out = np.array(pil_img)

    # Sørg for at labels kommer tilbage som (H, W, 1)
    if is_label and out.ndim == 2:
        out = out[:, :, np.newaxis]

    return out


class RescaleT(object):
    """
    Resize både billede og label til en given størrelse (kvadratisk).
    Forventet input:
        sample["image"]: numpy (H, W, 3)
        sample["label"]: numpy (H, W) eller (H, W, 1)
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample["imidx"], sample["image"], sample["label"]

        new_h, new_w = self.output_size

        img = _pil_resize(image, new_h, new_w, is_label=False)
        lbl = _pil_resize(label, new_h, new_w, is_label=True)

        return {"imidx": imidx, "image": img, "label": lbl}


class ToTensorLab(object):
    """
    Konverter sample til PyTorch tensors.
    Vi bruger kun flag=0 (RGB) i vores kode.
    Output:
        image: (3, H, W), float32 i [0,1]
        label: (1, H, W), float32 i [0,1]
    """

    def __init__(self, flag=0):
        self.flag = flag  # gemt for kompatibilitet, men vi bruger kun 0

    def __call__(self, sample):
        imidx, image, label = sample["imidx"], sample["image"], sample["label"]

        # Normaliser label til [0,1]
        label = label.astype(np.float32)
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if np.max(label) > 1e-6:
            label = label / np.max(label)

        # Normaliser billede til [0,1] og ændr til (C,H,W)
        image = image.astype(np.float32) / 255.0
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # sikrer (H,W,C)

        # HWC -> CHW
        image = image.transpose((2, 0, 1))  # (C,H,W)
        label = label.transpose((2, 0, 1))  # (1,H,W)

        return {
            "imidx": torch.from_numpy(imidx),
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
        }
