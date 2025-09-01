import json

from datasets import load_dataset
from prompt import get_tool_description


def process_maze(item):
    tools = json.dumps(get_tool_description(), ensure_ascii=False)
    item.pop("tools")
    for message in item['messages']:
        if message['role'] == "tool_call":
            message['content'] = message['content'].replace(
                "excute_python_code_in_jupyter", "execute_python_code_in_jupyter"
            ).replace("python_code", "execute_python_code_in_jupyter")
    return {
        **item,
        "tools": tools,
    }

if __name__ == "__main__":
    ds = load_dataset("json", data_files="/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/O3_new_new/maze/random_rect_maze_4k/processed_all_data/maze_code_sft_swift.jsonl", split="train")
    print(ds)
    ds = ds.map(process_maze, num_proc=10, remove_columns=ds.column_names)
    print(ds)
    ds.to_json("/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_sft_data/maze_code.jsonl", num_proc=10)

    for key in ds[0]:
        if key == "images":
            print(key, len(ds[0][key]))
        else:
            print(key, ds[0][key])
        print("-" * 100)