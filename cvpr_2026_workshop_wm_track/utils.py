from PIL import Image
import numpy as np
import math
from typing import Sequence, Any
from itertools import accumulate



def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL.
    Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """

    def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
        cur_width, cur_height = image.size
        if cur_width == width and cur_height == height:
            return image  # No need to resize if the image is already the correct size.

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_image = image.resize((resized_width, resized_height), resample=method)

        zero_image = Image.new(resized_image.mode, (width, height), 0)
        pad_height = max(0, int((height - resized_height) / 2))
        pad_width = max(0, int((width - resized_width) / 2))
        zero_image.paste(resized_image, (pad_width, pad_height))
        assert zero_image.size == (width, height)
        return zero_image

    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def split_data(data: Sequence[Any], world_size: int = 1, rank: int = 0) -> list[Any]:
    """Split a list-like dataset across ``world_size`` ranks.

    Args:
        data (Sequence[Any]): A list-like sequence to split.
        world_size (int): Total number of splits.
        rank (int): Current rank in [0, world_size).

    Returns:
        list[Any]: The split segment owned by ``rank``.
    """
    # Compute near-uniform splits with at-most-one element difference
    data_size = len(data)
    local_size = math.floor(data_size / world_size)
    local_size_list = [local_size for _ in range(world_size)]
    for i in range(data_size - local_size * world_size):
        local_size_list[i] += 1
    assert sum(local_size_list) == data_size
    local_size_list = [0] + list(accumulate(local_size_list))
    begin = local_size_list[rank]
    end = local_size_list[rank + 1]
    data = data[begin:end]
    return data