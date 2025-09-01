import json
import os
import uuid
from io import BytesIO

from datasets import load_dataset
from PIL import Image
from prompt import query_template
from qwen_vl_utils import smart_resize

MAX_PIXELS = 4096 * 2 * 28 * 28

data_file = "/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/TreeVGR-RL-37K/vstar30k_visdrone6k_x1y1x2y2.parquet"
image_dir = "/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/TreeVGR-RL-37K/"




"""
{'images': ['images/0.jpg'], 
'problem': '<image>\nIs the snowboard to the left or right of the skis?', 
'answer': 'The snowboard is to the right of the skis.', 
'target_instances': [
{'bbox': [264.4, 542.0, 300.46, 547.79], 'label': None, 'name': 'snowboard'}, 
{'bbox': [40.31, 375.0, 50.980000000000004, 375.91], 'label': None, 'name': 'skis'}
]
}
"""
def process_item_tree_rl(item):
    new_item = {
        "data_source": "TreeVGR-RL",
    }
    assert len(item["images"]) == 1, "only one image is supported"
    image_path = os.path.join(image_dir, item["images"][0])

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image = {
        "bytes": image_bytes,
        "path": image_path,   # 可选：如果你不打算用原始路径，设 None 也行
    }
    pil_image = Image.open(BytesIO(image_bytes))
    image_id = uuid.uuid4().hex
    new_item["images"] = [image]
    width, height = pil_image.size
    prompts = [
        {
            "role": "user",
            "content": query_template(item["problem"].replace("<image>", "").strip(), 
            f"./{image_id}.jpg"),
        },
    ]
    new_item['image_id'] = [image_id]
    resized_height, resized_width = smart_resize(
        height=height, width=width, max_pixels=MAX_PIXELS
    )

    new_item["prompt"] = prompts

    new_item["reward_model"] = {
        "ground_truth": item["answer"],
        "style": "model",
    }


    bbox_list = json.dumps(item["target_instances"], ensure_ascii=False)
    
    new_item["abilities"] = "Detail-Perception"
    new_item["extra_info"] = {
        "question": item["problem"],
        "answer": item["answer"],
        "bbox": bbox_list,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "ori_width": width,
        "ori_height": height,
    }

    return new_item

def filter_large_image(item):
    w, h = item['extra_info']["ori_width"], item['extra_info']["ori_height"]
    if w * h > 6000 * 28 * 28:
        return False
    return True



if __name__ == "__main__":
    ds = load_dataset("parquet", data_files=data_file)["train"]
    print(ds)

    MAX_PIXELS = 28 * 28 * 4096 * 2


    ds = ds.map(process_item_tree_rl, num_proc=16, remove_columns=ds.column_names)
    print(ds)
    ds = ds.filter(filter_large_image, num_proc=16)
    print(ds)
    save_path = f"/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_train/tree_rl_{len(ds)//1000}k.parquet"
    ds.to_parquet(save_path)
    print(ds)
    for key in ds[0]:
        if key == "images":
            continue
        print(key, ds[0][key])

    print(f"save to {save_path}")