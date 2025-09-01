import json
import uuid

from datasets import load_dataset
from prompt import get_tool_description, query_template


def process_item(item):

    item['tools'] = json.dumps(get_tool_description(), ensure_ascii=False)
    question = item['question'].replace("<image>", "").strip()
    image_id = uuid.uuid4().hex
    sandbox_image_path = f"./{image_id}.jpg"

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
        "image_id": [image_id],
        "images": item['images']
    }

if __name__ == "__main__":
    data_path = "/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/process_shk/image_math_code/add_answer_tag/tmp/results_0_filter_calculation_only_new_prompt.jsonl"
    ds = load_dataset("json", data_files=data_path, split="train")
    print(ds)

    ds = ds.map(process_item, num_proc=10)
    print(ds)

    ds.to_json("/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_sft_data/mm_math.jsonl", num_proc=10)

    for key in ds[0]:
        if key == "images":
            print(key, len(ds[0][key]))
        else:
            print(key, ds[0][key])
        print("-"*100)