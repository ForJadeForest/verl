import json
import uuid

from datasets import load_dataset
from prompt import get_tool_description, query_template


def get_img_id(text):
    """
 have upload the following images:\nPicture 0 path: \"./b4e09df7-4211-4d01-88dd-4f243c606a56.jpg\"\n\n\nNow please answer the folloing question.
    """
    import re

    pattern = r"Picture \d+ path: \"(.*)\""
    match = re.search(pattern, text)
    if match:
        image_path = match.group(1)
        image_id = image_path.split("/")[-1].split(".")[0]
        return image_id
    else:
        raise ValueError(f"No image path found in text: {text}")

def process_item(item):
    item['tools'] = json.dumps(get_tool_description(), ensure_ascii=False)
    question = item['question'].replace("<image>", "").strip()
    old_img_ids = get_img_id(item['messages'][1]['content'])
    
    sandbox_image_path = f"./{old_img_ids}.jpg"

    query = query_template(question, sandbox_image_path)
    new_messages = item['messages']
    new_messages[1]['content'] = query

    for message in new_messages:
        if message['role'] == "tool_call":
            message['content'] = message['content'].replace(
                "excute_python_code_in_jupyter", "execute_python_code_in_jupyter"
            )

    assert query.count("<image>") == 1, f"query: {query}"
    
    return {
        "messages": new_messages,
        "tools": item['tools'],
        "question": item['question'],
        "answer": item['answer'],
        "source": item['source'],
        "image_id": [old_img_ids],
        "images": item['images']
    }

if __name__ == "__main__":
    data_path = "/apdcephfs_gy5/share_303588738/yingzhepeng/results/O3_data/english/cot_generate_53k_v3/train_data_tool_desc_v2/converted_code_check_data.jsonl"
    ds = load_dataset("json", data_files=data_path, split="train")
    print(ds)

    ds = ds.map(process_item, num_proc=10)
    print(ds)

    ds.to_json("/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_sft_data/mm_crop.jsonl", num_proc=10)

    for key in ds[0]:
        if key == "images":
            print(key, len(ds[0][key]))
        else:
            print(key, ds[0][key])
    