import base64
import io
import mmap
import os
import random
import string
from io import BytesIO
from typing import Union

from PIL import Image
from qwen_vl_utils import smart_resize

from evaluate.infer_engine_utils import IMAGE_FACTOR, MAX_PIXELS, MIN_PIXELS


def read_patch_image(patch_path: str, start_pos: int, read_size: int) -> Image.Image:
    """Read image from patch file"""

    # Read image data
    with open(patch_path, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            mm.seek(start_pos)
            image_data = mm.read(read_size)

    # Parse image
    image = Image.open(io.BytesIO(image_data))
    return image


def generate_random_id(length: int = 10) -> str:
    """Generate a random id"""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def guess_image_from_item(item) -> Image.Image:
    """Guess image from item
    Supported keys:
    - decoded_image
    - original_image_path
    - image_path
    - image_patch_info
    """
    if "decoded_image" in item and isinstance(item["decoded_image"], Image.Image):
        image = item["decoded_image"]
    elif "original_image_path" in item and isinstance(item["original_image_path"], str) and os.path.exists(item["original_image_path"]):
        image = Image.open(item["original_image_path"]).convert("RGB")
    elif "image_path" in item and isinstance(item["image_path"], str) and os.path.exists(item["image_path"]):
        image = Image.open(item["image_path"]).convert("RGB")
    elif "image_patch_info" in item:
        image = read_patch_image(
            item["image_patch_info"]["patch"],
            item["image_patch_info"]["start_num"],
            item["image_patch_info"]["size"],
        )
    else:
        raise ValueError(
            "Item must contain 'decoded_image', 'original_image_path', 'image_path', or 'image_patch_info'"
        )

    return image


def encode_image_path_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def encode_image_base64(image: Union[str, Image.Image]) -> str:
    """Encode image to base64 string"""
    if isinstance(image, str):
        return encode_image_path_base64(image)
    elif isinstance(image, Image.Image):
        return encode_pil_image_to_base64(image)
    else:
        raise ValueError("Image must be a file path or PIL Image instance")


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")


def qwen_resize_image(
    image: Image.Image,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Image.Image:

    w, h = image.size
    resized_h, resized_w = smart_resize(h, w, factor, min_pixels, max_pixels)
    image = image.resize((resized_w, resized_h))
    return image
