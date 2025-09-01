"""
VERL dataset format:
1. data_source: "Ground-R1"
2. prompt: list[dict] # openai message format
3. images: list[dict] # a list of Image
4. abilities: "ground"
6. image_id: str # uuid
7. reward_model: dict {"ground_truth": str, "style": model}
8. extra_info: dict {"question": str, "answer": str, "bbox": list[float]}
"""

import json
import uuid
from io import BytesIO

from datasets import load_dataset
from PIL import Image
from prompt import get_system_prompt, query_template
from qwen_vl_utils import smart_resize


def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """
    把 PIL.Image 转为无损 PNG 二进制；你也可以改成 JPEG 等格式。
    """
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def process_item_ground_r1(item):
    bboxs = item["bboxs"]

    width, height = item["width"], item["height"]
    assert len(bboxs) == 1, "only one bbox is supported"

    image_paths = item["image"]
    if image_paths.startswith("path/to/"):
        image_paths = image_paths.replace(
            "path/to/", 
            "/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/Ground-R1/", 
            1
        )

    with open(image_paths, "rb") as f:
        image_bytes = f.read()
    pil_image = Image.open(BytesIO(image_bytes))
    image = {
        "bytes": image_bytes,
        "path": image_paths,   # 可选：如果你不打算用原始路径，设 None 也行
    }
    width, height = pil_image.size
    resized_height, resized_width = smart_resize(
        height=height, width=width, max_pixels=MAX_PIXELS
    )

    image_id = uuid.uuid4().hex
    assert item["solution"], f"solution is None: {item}"
    assert item["problem"], f"problem is None: {item}"
    prompts = [
        {
            "role": "user",
            "content": query_template(item["problem"].replace("<image>", ""), f"./{image_id}.jpg"),
        },
    ]
    assert prompts[0]["content"].count("<image>") == 1, f"image count is not 1: {prompts[0]['content']}"
    new_item = {
        "data_source": "Ground-R1",
        "prompt": prompts,
        "images": [image],
        "abilities": "ground",
        "image_id": [image_id],
        "reward_model": {"ground_truth": item["solution"], "style": "model"},
        "extra_info": {
            "question": item["problem"],
            "answer": item["solution"],
            "bbox": bboxs[0],
            "resized_width": resized_width,
            "resized_height": resized_height,
            "ori_width": width,
            "ori_height": height,
        },
    }
    return new_item

def filter_large_image(item):
    w, h = item['extra_info']["ori_width"], item['extra_info']["ori_height"]
    if w * h > 6000 * 28 * 28:
        return False
    return True

if __name__ == "__main__":
    ds = load_dataset("json", data_files="/mnt/private_yingzhepeng/code/rl/DeepEyes/data/Ground-R1/train_33k.jsonl")["train"]
    print(ds)

    MAX_PIXELS = 28 * 28 * 4096 * 2


    ds = ds.map(process_item_ground_r1, num_proc=16, remove_columns=ds.column_names)
    print(ds)
    ds = ds.filter(filter_large_image, num_proc=16)
    print(ds)
    
    ds.to_parquet(f"/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_train/ground_r1_{len(ds)//1000}k.parquet")
    print(ds)
    for key in ds[0]:
        if key == "images":
            continue
        print(key, ds[0][key])
