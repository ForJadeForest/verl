import json
import uuid

import datasets
from datasets import load_dataset
from prompt import query_template

base_dir = (
    "/apdcephfs_sh8/share_301266059/berlinni/shihoukun/deepeyes/DeepEyes-Datasets-47k"
)

data_file = [
    f"{base_dir}/data_0.1.2_visual_toolbox_v2.parquet",
    f"{base_dir}/data_v0.8_visual_toolbox_v2.parquet",
    f"{base_dir}/data_thinklite_reasoning_acc.parquet",
]
all_ds = []

for file in data_file:
    ds = load_dataset("parquet", data_files=file, split="train")
    all_ds.append(ds)

all_ds = datasets.concatenate_datasets(all_ds)


def process(item):
    question = item['extra_info']['question']
    item_id = uuid.uuid4().hex
    sandbox_image_path = [f"./{item_id}.jpg"]
    query = query_template(question, sandbox_image_path)
    prompt = [
        {"role": "user", "content": query}
    ]
    answer = item['extra_info']['answer']
    images = item["images"] # a list of dict with "bytes" and "path"
    

    if item["data_source"] == "thinklite_eureka":
        prompt[0]["content"] = prompt[0]["content"] + "\nYou should put your answer in \\boxed{...}\n"
    return {
        "data_source": item["data_source"],
        "prompt": prompt,
        "images": images,
        "abilities": item["ability"],
        "image_id": [item_id],
        "reward_model": item['reward_model'],
        "extra_info": {"question": question, "answer": answer, "index": item_id},
    }


ds: datasets.Dataset = all_ds.map(process, num_proc=16, remove_columns=all_ds.column_names)

train_ds = ds.train_test_split(test_size=0.1)
# train_ds.to_parquet(
#     f"/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_train/deepeyes_train_{len(train_ds)//1000}k.parquet"
# )
# train_ds["test"].to_parquet(
#     f"/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_train/temp_deepeyes_val_{len(train_ds['test'])//1000}k.parquet"
# )
ds.to_parquet(
    f"/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_train/deepeyes_train_{len(ds)//1000}k.parquet"
)

math_ds = ds.filter(lambda x: x["data_source"] == "thinklite_eureka")
math_ds.to_parquet(
    f"/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_train/deepeyes_train_math_{len(math_ds)//1000}k.parquet"
)
non_math_ds = ds.filter(lambda x: x["data_source"] != "thinklite_eureka")
non_math_ds.to_parquet(
    f"/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_train/deepeyes_train_non_math_{len(non_math_ds)//1000}k.parquet"
)

for key in ds[0]:
    if key == "images":
        print(type(ds[0][key]))
        continue
    print(key, ds[0][key])
