import json
import re
import uuid

from datasets import load_dataset
from prompt import get_tool_description, query_template


def get_question(text):
    return (
        text.split("he folloing question.\nQuestion:")[-1]
        .split("You should write the answer in the following format:\n```plaintext\n")[0]
        .strip()
    )


def get_image_path(text):
    image_paths = re.findall(r"Picture \d+ path: \"(.*)\"", text)
    assert len(image_paths) == 1, f"image_paths: {image_paths}"
    return image_paths[0]


def process_thyme(item):
    question = get_question(item['messages'][1]['content']).replace("<image>", "")
    new_messages = item["messages"]
    new_messages[0]["content"] = "You are a helpful assistant."

    image_path = get_image_path(item['messages'][1]['content'])
    new_messages[1]["content"] = query_template(question, [image_path])
    image_ids = [uuid.uuid4().hex]
    images = []
    for message in new_messages:
        if message['role'] == "tool_call":
            message['content'] = message['content'].replace(
                "excute_python_code_in_jupyter", "execute_python_code_in_jupyter"
            )
    for image in item['images']:
        if not image.startswith("data:image/"):
            images.append(f"data:image/jpeg;base64,{image}")
        else:
            images.append(image)
    return {
        "messages": new_messages,
        "image_id": image_ids,
        "images": images,
        "tools": json.dumps(get_tool_description(), ensure_ascii=False),
        "original_question": item["original_question"],
        "original_response": item["original_response"],
    }

def process_thyme_nocode(item):
    question = get_question(item['message'][1]['content']).replace("<image>", "")
    new_messages = item["message"]
    new_messages[0]["content"] = "You are a helpful assistant."

    image_path = get_image_path(item['message'][1]['content'])
    new_messages[1]["content"] = query_template(question, [image_path])
    for message in new_messages:
        if message['role'] == "tool_call":
            message['content'] = message['content'].replace(
                "excute_python_code_in_jupyter", "execute_python_code_in_jupyter"
            )
    image_ids = [uuid.uuid4().hex]
    images = []

    for image in item['images']:
        if not image.startswith("data:image/"):
            images.append(f"data:image/jpeg;base64,{image}")
        else:
            images.append(image)
    return {
        "messages": new_messages,
        "image_id": image_ids,  
        "images": images,
        "tools": json.dumps(get_tool_description(), ensure_ascii=False),
        "meta_info": {**item["meta_info"], "response": item['response'], "question": question},
    }

if __name__ == "__main__":
    import pyarrow.parquet as pq
    from datasets import Dataset

    path = "/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/Thyme-Processed/thyme_code_split.parquet"

    # 读成 Arrow Table
    table = pq.read_table(path)

    # 删除 schema 上的 metadata（尤其是 'huggingface' 键）
    table = table.replace_schema_metadata(None)

    # 包成 HF Dataset（不从 HF metadata 恢复特征）
    ds = Dataset(table)
    print(ds)

    # ds = load_dataset("parquet", data_files="/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/Thyme-Processed/thyme_code_split.parquet", split="train")
    print(ds)
    ds = ds.map(process_thyme, num_proc=16, remove_columns=ds.column_names)
    print(ds)

    ds.to_parquet("/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_sft_data/thyme_code.parquet")
    

    for key in ds[0]:
        if key == "images":
            print(key, len(ds[0][key]))
        else:
            print(key, ds[0][key])
        print("-" * 100)

    
    ds = load_dataset("json", data_files="/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/Thyme-Processed/thyme_nocode_split_msswift_sft.jsonl", split="train")
    print(ds)
    ds = ds.map(process_thyme_nocode, num_proc=10, remove_columns=ds.column_names)
    print(ds)
    ds.to_parquet("/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_sft_data/thyme_nocode.parquet")

    for key in ds[0]:
        if key == "images":
            print(key, len(ds[0][key]))
        else:
            print(key, ds[0][key])
        print("-" * 100)