import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
import numpy as np
from PIL import Image


def read_dcm(path):
    dcm = pydicom.read_file(path)
    img = apply_voi_lut(apply_modality_lut(dcm.pixel_array, dcm), dcm)
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    img = np.invert(img)
    if len(img.shape) < 3:
        img = np.repeat(img[..., None], 3, axis=-1)
    return img


def openImage(path):
    if (path.endswith("png") or path.endswith("jpeg")):
        return Image.open(path)
    else:
        file = read_dcm(path)
        file = Image.fromarray(file)
        return file.convert("L")
