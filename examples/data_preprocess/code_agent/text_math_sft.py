import json
import uuid

from datasets import load_dataset
from prompt import get_tool_description, query_template
from transformers import AutoTokenizer


def process_item(item):

    item['tools'] = json.dumps(get_tool_description(), ensure_ascii=False)
    for message in item['messages']:
        if message['role'] == "tool_call":
            message['content'] = message['content'].replace(
                "excute_python_code_in_jupyter", "execute_python_code_in_jupyter"
            )
    return item

from swift.llm import get_model_tokenizer, get_template

_, tokenizer = get_model_tokenizer('Qwen/Qwen2.5-VL-7B-Instruct', load_model=False)
template = get_template(tokenizer.model_meta.template, tokenizer, agent_template='hermes', max_length=327680000)

def filter_item(item):
    encoded = template.encode(item)
    if len(encoded["input_ids"]) > 32768:
        return False
    return True


if __name__ == "__main__":
    data_path = "/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/process_shk/nvidia_math_code_filter_30k/convert_data/correct_v1_10k_code_check_add_tag.jsonl"
    ds = load_dataset("json", data_files=data_path, split="train")
    print(ds)

    ds = ds.map(process_item, num_proc=10)
    print(ds)
    ds = ds.filter(filter_item, num_proc=10)
    print(ds)
    ds.to_json("/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_sft_data/text_math.jsonl", num_proc=10)

    for key in ds[0]:
        if key == "images":
            print(key, len(ds[0][key]))
        else:
            print(key, ds[0][key])
    