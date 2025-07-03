import cv2
import numpy as np
from PIL import Image


def tensor_to_disparity_image(tensor_data):
    assert len(tensor_data.size()) == 2
    assert (tensor_data >= 0.0).all().item()

    disparity_image = Image.fromarray(np.asarray(tensor_data * 256.0).astype(np.uint16))

    return disparity_image


def tensor_to_disparity_magma_image(tensor_data, vmax=None, mask=None):
    assert len(tensor_data.size()) == 2
    assert (tensor_data >= 0.0).all().item()

    numpy_data = np.asarray(tensor_data)

    if vmax is not None:
        numpy_data = numpy_data * 255 / vmax
        numpy_data = np.clip(numpy_data, 0, 255)

    numpy_data = numpy_data.astype(np.uint8)
    numpy_data_magma = cv2.applyColorMap(numpy_data, cv2.COLORMAP_MAGMA)  # JET  RAINBOW
    numpy_data_magma = cv2.cvtColor(numpy_data_magma, cv2.COLOR_BGR2RGB)

    if mask is not None:
        assert tensor_data.size() == mask.size()
        numpy_data_magma[~mask] = [255, 255, 255]

    disparity_image = Image.fromarray(numpy_data_magma)

    return disparity_image

def tensor_to_disparity_jet_image(tensor_data, vmax=None, mask=None):
    assert len(tensor_data.size()) == 2
    assert (tensor_data >= 0.0).all().item()

    tensor_data[tensor_data == float('inf')] = 0
    numpy_data = np.asarray(tensor_data)
    if vmax is not None:
        numpy_data = numpy_data * 255 / vmax
        numpy_data = np.clip(numpy_data, 0, 255)
    numpy_data = numpy_data.astype(np.uint8)
    numpy_data_magma = cv2.applyColorMap(numpy_data, cv2.COLORMAP_JET)  # JET  RAINBOW
    numpy_data_magma = cv2.cvtColor(numpy_data_magma, cv2.COLOR_BGR2RGB)

    if mask is not None:
        assert tensor_data.size() == mask.size()
        numpy_data_magma[~mask] = [255, 255, 255]

    disparity_image = Image.fromarray(numpy_data_magma)

    return disparity_image

def nparray_to_disparity_magma_image(numpy_data, vmax=None, mask=None):
    assert len(numpy_data.size()) == 2
    assert (numpy_data >= 0.0).all().item()

    if vmax is not None:
        numpy_data = numpy_data * 255 / vmax
        numpy_data = np.clip(numpy_data, 0, 255)

    numpy_data = numpy_data.astype(np.uint8)
    numpy_data_magma = cv2.applyColorMap(numpy_data, cv2.COLORMAP_MAGMA)  # JET  RAINBOW
    numpy_data_magma = cv2.cvtColor(numpy_data_magma, cv2.COLOR_BGR2RGB)

    if mask is not None:
        assert numpy_data.size() == mask.size()
        numpy_data_magma[~mask] = [255, 255, 255]

    disparity_image = Image.fromarray(numpy_data_magma)

    return disparity_image

# def _make_disparity_image(disparity, height, width):
#     DISPARITY_MULTIPLIER = 7.0
#     INVALID_DISPARITY = 255
#     disparity_image = np.full(
#         (height, width),
#         disparity * DISPARITY_MULTIPLIER,
#         dtype=np.int)
#     disparity_image = np.clip(disparity_image, a_min=None, a_max=254)
#     invalid_mask = (disparity == 0)
#     disparity_image[invalid_mask] = INVALID_DISPARITY
#     return disparity_image

# def _save_disparity_image(filename, disparity_image):
#     Image.fromarray(disparity_image.astype(np.uint8)).save(filename)
