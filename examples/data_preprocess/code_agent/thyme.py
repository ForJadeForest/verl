from datasets import load_dataset
from prompt import query_template
import uuid

ds = load_dataset("Kwai-Keye/Thyme-RL", split="train")

def process_thyme(item):
    question = item.pop("question")
    answer = item.pop("solution")
    image_id = uuid.uuid4().hex

    images = item.pop("images") # base64 encoded images
    images = ["data:image/jpeg;base64," + image for image in images]
    
    prompt = [
        {"role": "user", "content": query_template(question, [f"./{image_id}.jpg"])}
    ]

    return {
        "data_source": "Thyme-RL",
        "prompt": prompt,
        "images": images,
        "abilities": "general",
        "image_id": [image_id],
        "reward_model": {"ground_truth": answer, "style": "model"},
        "extra_info": {
            "question": question,
            "answer": answer,
        }
    }

ds = ds.map(process_thyme, num_proc=10, remove_columns=ds.column_names)
ds.to_parquet(f"/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_train/thyme_{len(ds)//1000}k.parquet")