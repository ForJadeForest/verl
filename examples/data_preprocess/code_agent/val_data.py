import uuid
from functools import partial
import base64
import io
import os
import json
import re

# Disable HF datasets caching to avoid corrupted image issues
# os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
# os.environ["HF_DATASETS_OFFLINE"] = "0"
# os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"

from datasets import load_dataset
from PIL import Image
from prompt import query_template

Image.MAX_IMAGE_PIXELS = 500000000

def image_to_base64(image):
    """
    Convert PIL Image to base64 string
    
    Args:
        image: PIL Image object or base64 string
        
    Returns:
        base64 string with data URL prefix
    """
    if isinstance(image, str):
        # If already base64, just add prefix if needed
        if image.startswith('data:image/'):
            return image
        else:
            return f"data:image/jpeg;base64,{image}"
    
    # Convert PIL Image to base64
    buffer = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffer, format='JPEG', quality=95)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

"""
verl val data build:
VERL dataset format:
1. data_source: "xxx"
2. prompt: list[dict] # openai message format
3. images: list[str] # a list of base64 encoded images with data URL prefix
4. abilities: "xxx"
5. image_id: str # uuid
6. reward_model: dict {"ground_truth": str, "style": model}
7. extra_info: dict {"question": str, "answer": str, "bbox": list[float]}

Note: All images are converted to base64 format with "data:image/jpeg;base64," prefix
for consistency across different datasets.
"""

print("=" * 80)
print("🔍 LOADING MMStar DATASET")
print("=" * 80)
mm_star_ds = load_dataset("Lin-Chen/MMStar", split="val")


def process_mm_star(item):
    question = item["question"]
    answer = item["answer"]
    image = item["image"]
    image_id = uuid.uuid4().hex
    sandbox_image_path = [f"./{image_id}.jpg"]

    query = query_template(question=question, sandbox_image_path=sandbox_image_path, need_picture_text=True)
    return {    
        "data_source": "MMStar",
        "prompt": [
            {"role": "user", "content": query},
        ],
        "images": [image_to_base64(image)],
        "abilities": "general",
        "image_id": [image_id],
        "sandbox_image_path": sandbox_image_path,
        "reward_model": {"ground_truth": answer, "style": "model"},
        "extra_info": {
            "question": question,
            "answer": answer,
            "metadata": json.dumps({
                "meta_info": item["meta_info"],
                "category": item["category"],
                "l2_category": item["l2_category"],
                "index": item["index"],
            }, ensure_ascii=False),
        },
    }


print("=" * 80)
print("🔍 PROCESSING MMStar DATASET")
print("=" * 80)
mm_star_ds = mm_star_ds.map(process_mm_star, num_proc=10, remove_columns=mm_star_ds.column_names)

print("=" * 80)
print("🔍 LOADING MathVista DATASET")
print("=" * 80)
mathvista_ds = load_dataset("AI4Math/MathVista", split="testmini")

print("=" * 80)
print("🔍 PROCESSING MathVista DATASET")
print("=" * 80)
def process_mathvista(item):
    question = item.pop("query")
    answer = item.pop("answer")
    image = item.pop("decoded_image")
    image_id = uuid.uuid4().hex
    sandbox_image_path = [f"./{image_id}.jpg"]
    return {
        "data_source": "MathVista",
        "prompt": [
            {"role": "user", "content": query_template(question=question, sandbox_image_path=sandbox_image_path)},
        ],
        "images": [image_to_base64(image)],
        "abilities": "general",
        "image_id": [image_id],
        "sandbox_image_path": sandbox_image_path,
        "reward_model": {"ground_truth": answer, "style": "model"},
        "extra_info": {"question": question, "answer": answer, "metadata": json.dumps({**item}, ensure_ascii=False)},
    }


mathvista_ds = mathvista_ds.map(process_mathvista, num_proc=10, remove_columns=mathvista_ds.column_names)


print("=" * 80)
print("🔍 LOADING MathVerse DATASET")
print("=" * 80)
mathverse_ds = load_dataset("AI4Math/MathVerse", "testmini", split="testmini")

print("=" * 80)
print("🔍 PROCESSING MathVerse DATASET")
print("=" * 80)
def process_mathverse(item):
    question = item.pop("question")
    answer = item.pop("answer")
    image = item.pop("image")
    image_id = uuid.uuid4().hex
    sandbox_image_path = [f"./{image_id}.jpg"]
    return {
        "data_source": "MathVerse",
        "prompt": [
            {"role": "user", "content": query_template(question=question, sandbox_image_path=sandbox_image_path)},
        ],
        "images": [image_to_base64(image)],
        "abilities": "mm_math",
        "image_id": [image_id],
        "sandbox_image_path": sandbox_image_path,
        "reward_model": {"ground_truth": answer, "style": "model"},
        "extra_info": {"question": item["question_for_eval"], "answer": answer, "metadata": json.dumps({**item}, ensure_ascii=False)},
    }


mathverse_ds = mathverse_ds.map(process_mathverse, num_proc=10, remove_columns=mathverse_ds.column_names)

# print("=" * 80)
# print("🔍 LOADING We-Math2.0-Standard DATASET")
# print("=" * 80)
# wemath2_ds = load_dataset("We-Math/We-Math2.0-Standard", split="standard")


# def process_wemath2(item):
#     question = item.pop("question")
#     answer = item.pop("answer")
#     image = item.pop("image")
#     image_id = uuid.uuid4().hex
#     sandbox_image_path = [f"./{image_id}.jpg"]
#     return {
#         "data_source": "We-Math2.0-Standard",
#         "prompt": [
#             {"role": "user", "content": query_template(question=question, sandbox_image_path=sandbox_image_path)},
#         ],
#         "images": [image_to_base64(image)],
#         "abilities": "mm_math",
#         "image_id": [image_id],
#         "sandbox_image_path": sandbox_image_path,
#         "reward_model": {"ground_truth": answer, "style": "model"},
#         "extra_info": {"question": question, "answer": answer, "metadata": {**item}},
#     }


# wemath2_ds = wemath2_ds.map(process_wemath2, num_proc=10, remove_columns=wemath2_ds.column_names)

print("=" * 80)
print("🔍 LOADING HallusionBench DATASET")
print("=" * 80)
hallusion_bench_ds = load_dataset("lmms-lab/HallusionBench", split="image")

print("=" * 80)
print("🔍 PROCESSING HallusionBench DATASET")
print("=" * 80)
def process_hallusion_bench(item):
    question = item.pop("question") + "\nPlease answer yes or no."
    answer = item.pop("gt_answer")
    if int(answer) == 1:
        answer = "yes"
    else:
        answer = "no"
    image = item.pop("image")
    image_id = uuid.uuid4().hex
    sandbox_image_path = [f"./{image_id}.jpg"]
    return {
        "data_source": "HallusionBench",
        "prompt": [
            {"role": "user", "content": query_template(question=question, sandbox_image_path=sandbox_image_path)},
        ],
        "images": [image_to_base64(image)],
        "abilities": "Detail-Perception",
        "image_id": [image_id],
        "sandbox_image_path": sandbox_image_path,
        "reward_model": {
            "ground_truth": answer,
            "style": "model",
        },
        "extra_info": {"question": question, "answer": answer, "metadata": json.dumps({**item}, ensure_ascii=False)},
    }


hallusion_bench_ds = hallusion_bench_ds.map(
    process_hallusion_bench, num_proc=10, remove_columns=hallusion_bench_ds.column_names
)


print("=" * 80)
print("🔍 LOADING MMMU DATASET")
print("=" * 80)
mmmu_ds = load_dataset("lmms-lab/MMMU", split="validation")

print("=" * 80)
print("🔍 PROCESSING MMMU DATASET")
print("=" * 80)
def process_mmmu(item):
    question = item.pop("question")
    options = item.pop("options")
    options_char = [chr(i) for i in range(65, 65 + len(options))]
    options_str = "\n".join([f"{char}: {option}" for char, option in zip(options_char, options)])
    question = f"{question}\n{options_str}"
    answer = item.pop("answer")
    images = []
    sandbox_image_path = []
    image_ids = []
    
    for i in range(1, 8):
        image_column = f"image_{i}"
        image = item.pop(image_column)
        if image is None:
            continue
        if f"<image {i}>" not in question:
            continue
        image_id = uuid.uuid4().hex
        sandbox_image_path.append(f"./{image_id}.jpg")
        images.append(image_to_base64(image))
        image_ids.append(image_id)

    query = query_template(question=question, sandbox_image_path=sandbox_image_path, need_picture_text=False)
    for i in range(1, 8):
        if query.count(f"<image {i}>") == 1:
            query = query.replace(f"<image {i}>", f"Picture {i}: <image>")
        else:
            query = query_template(question=question, sandbox_image_path=sandbox_image_path, need_picture_text=True)

    prompt = [
        {"role": "user", "content": query}
    ]

    return {
        "data_source": "MMMU",
        "prompt": prompt,
        "images": images,
        "abilities": "general",
        "image_id": image_ids,
        "sandbox_image_path": sandbox_image_path,
        "reward_model": {"ground_truth": answer, "style": "model"},
        "extra_info": {"question": question, "answer": answer, "metadata": json.dumps({**item}, ensure_ascii=False)},
    }


mmmu_ds = mmmu_ds.map(process_mmmu, num_proc=10, remove_columns=mmmu_ds.column_names)


print("=" * 80)
print("🔍 LOADING VStar DATASET")
print("=" * 80)
vstar_ds = load_dataset("lmms-lab/vstar-bench", split="test")

print("=" * 80)
print("🔍 PROCESSING VStar DATASET")
print("=" * 80)
def process_vstar(item):
    question = item.pop("text")
    answer = item.pop("label")
    image = item.pop("image")
    image_id = uuid.uuid4().hex
    sandbox_image_path = [f"./{image_id}.jpg"]
    prompt = [
        {"role": "user", "content": query_template(question=question, sandbox_image_path=sandbox_image_path)}
    ]
    return {
        "data_source": "VStar",
        "prompt": prompt,
        "images": [image_to_base64(image)],
        "abilities": "Detail-Perception",
        "image_id": [image_id],
        "sandbox_image_path": sandbox_image_path,
        "reward_model": {"ground_truth": answer, "style": "model"},
        "extra_info": {"question": question, "answer": answer, "metadata": json.dumps({**item}, ensure_ascii=False)},
    }
vstar_ds = vstar_ds.map(process_vstar, num_proc=10, remove_columns=vstar_ds.column_names)





print("=" * 80)
print("🔍 LOADING AI2D DATASET")
print("=" * 80)
ai2d_ds = load_dataset("lmms-lab/ai2d", split="test")

print("=" * 80)
print("🔍 PROCESSING AI2D DATASET")
print("=" * 80)
def process_ai2d(item):
    question = item.pop("question")
    options = item.pop("options")
    options_char = [chr(i) for i in range(65, 65 + len(options))]
    options_str = "\n".join([f"{char}: {option}" for char, option in zip(options_char, options)])
    question = f"{question}\n{options_str}"
    answer = options_char[int(item.pop("answer"))]

    image = item.pop("image")
    image_id = uuid.uuid4().hex
    sandbox_image_path = [f"./{image_id}.jpg"]
    prompt = [
        {"role": "user", "content": query_template(question=question, sandbox_image_path=sandbox_image_path)}
    ]
    return {
        "data_source": "AI2D",
        "prompt": prompt,
        "images": [image_to_base64(image)],
        "abilities": "general",
        "image_id": [image_id],
        "sandbox_image_path": sandbox_image_path,
        "reward_model": {"ground_truth": answer, "style": "model"},
        "extra_info": {"question": question, "answer": answer, "metadata": json.dumps({**item}, ensure_ascii=False)},
    }


ai2d_ds = ai2d_ds.map(process_ai2d, num_proc=10, remove_columns=ai2d_ds.column_names)


hrbench_4k_ds = load_dataset("DreamMr/HR-Bench", split="hrbench_4k")
hrbench_8k_ds = load_dataset("DreamMr/HR-Bench", split="hrbench_8k")


def process_hrbench(item, split):
    question = item.pop("question")
    answer = item.pop("answer")
    image = item.pop("image") # is base64 encoded image
    image_id = uuid.uuid4().hex
    sandbox_image_path = [f"./{image_id}.jpg"]
    
    options_char = ["A", "B", "C", "D"]
    options = [item.pop(c) for c in options_char]
    options_str = "\n".join([f"{char}: {option}" for char, option in zip(options_char, options)])
    question = f"{question}\n{options_str}"

    prompt = [
        {"role": "user", "content": query_template(question=question, sandbox_image_path=sandbox_image_path)}
    ]
    return {
        "data_source": f"HRBench-{split}",
        "prompt": prompt,
        "images": [image_to_base64(image)],
        "abilities": "general",
        "image_id": [image_id],
        "sandbox_image_path": sandbox_image_path,
        "reward_model": {"ground_truth": answer, "style": "model"},
        "extra_info": {"question": question, "answer": answer, "metadata": json.dumps({**item}, ensure_ascii=False)},
    }

process_hrbench_4k = partial(process_hrbench, split="4k")
hrbench_4k_ds = hrbench_4k_ds.map(process_hrbench_4k, num_proc=10, remove_columns=hrbench_4k_ds.column_names)

process_hrbench_8k = partial(process_hrbench, split="8k")
hrbench_8k_ds = hrbench_8k_ds.map(process_hrbench_8k, num_proc=10, remove_columns=hrbench_8k_ds.column_names)


print("=" * 80)
print("🔍 LOADING TreeBench DATASET")
print("=" * 80)
tree_bench_ds = load_dataset("HaochenWang/TreeBench", split="train")

print("=" * 80)
print("🔍 PROCESSING TreeBench DATASET")
print("=" * 80)

def process_tree_bench(item):
    question = item.pop("question")
    options = item.pop("multi-choice options")

    question = f"{question}\n{options}"
    answer = item.pop("answer")
    image = item.pop("image") # is base64 encoded image
    image_id = uuid.uuid4().hex
    sandbox_image_path = [f"./{image_id}.jpg"]
    prompt = [
        {"role": "user", "content": query_template(question=question, sandbox_image_path=sandbox_image_path)}
    ]
    return {
        "data_source": "TreeBench",
        "prompt": prompt,
        "images": [image_to_base64(image)],
        "abilities": "Detail-Perception",
        "image_id": [image_id],
        "sandbox_image_path": sandbox_image_path,
        "reward_model": {"ground_truth": answer, "style": "model"},
        "extra_info": {"question": question, "answer": answer, "metadata": json.dumps({**item}, ensure_ascii=False)},
    }

tree_bench_ds = tree_bench_ds.map(process_tree_bench, num_proc=10, remove_columns=tree_bench_ds.column_names)

# %% Data sampling and concatenation
import random

from datasets import concatenate_datasets


def sample_dataset(dataset, max_samples=500, seed=42):
    """
    Sample dataset: if more than max_samples, randomly select max_samples; 
    otherwise keep unchanged.
    
    Args:
        dataset: HuggingFace dataset
        max_samples: maximum number of samples to keep
        seed: random seed for reproducibility
    
    Returns:
        Sampled dataset
    """
    if len(dataset) > max_samples:
        # Set random seed for reproducibility
        random.seed(seed)
        # Randomly select max_samples indices
        selected_indices = random.sample(range(len(dataset)), max_samples)
        # Select samples
        sampled_dataset = dataset.select(selected_indices)
        print(f"Dataset {dataset or 'Unknown'} sampled from {len(dataset)} to {len(sampled_dataset)} samples")
        return sampled_dataset
    else:
        print(f"Dataset {dataset or 'Unknown'} kept unchanged: {len(dataset)} samples")
        return dataset

# Sample all datasets
print("=" * 50)
print("Starting data sampling...")
print("=" * 50)

sampled_datasets = []

# Sample each dataset
mm_star_sampled = sample_dataset(mm_star_ds, max_samples=500, seed=42)
sampled_datasets.append(mm_star_sampled)

mathvista_sampled = sample_dataset(mathvista_ds, max_samples=500, seed=42)
sampled_datasets.append(mathvista_sampled)

mathverse_sampled = sample_dataset(mathverse_ds, max_samples=500, seed=42)
sampled_datasets.append(mathverse_sampled)

# wemath2_sampled = sample_dataset(wemath2_ds, max_samples=500, seed=42)
# sampled_datasets.append(wemath2_sampled)

hallusion_bench_sampled = sample_dataset(hallusion_bench_ds, max_samples=500, seed=42)
sampled_datasets.append(hallusion_bench_sampled)

mmmu_sampled = sample_dataset(mmmu_ds, max_samples=500, seed=42)
sampled_datasets.append(mmmu_sampled)

vstar_sampled = sample_dataset(vstar_ds, max_samples=500, seed=42)
sampled_datasets.append(vstar_sampled)

ai2d_sampled = sample_dataset(ai2d_ds, max_samples=500, seed=42)
sampled_datasets.append(ai2d_sampled)

hrbench_4k_sampled = sample_dataset(hrbench_4k_ds, max_samples=500, seed=42)
sampled_datasets.append(hrbench_4k_sampled)

hrbench_8k_sampled = sample_dataset(hrbench_8k_ds, max_samples=500, seed=42)
sampled_datasets.append(hrbench_8k_sampled)

tree_bench_sampled = sample_dataset(tree_bench_ds, max_samples=500, seed=42)
sampled_datasets.append(tree_bench_sampled)

# Concatenate all sampled datasets
print("=" * 50)
print("Concatenating all sampled datasets...")
print("=" * 50)

final_dataset = concatenate_datasets(sampled_datasets)

print(f"Final dataset size: {len(final_dataset)} samples")
print(f"Dataset features: {final_dataset.features}")

# %% Print sample from each dataset
print("\n" + "=" * 80)
print("📊 SAMPLE DATA FROM EACH DATASET")
print("=" * 80)

def print_dataset_sample(dataset, dataset_name, sample_index=0):
    """Print a beautiful sample from the dataset"""
    if len(dataset) == 0:
        print(f"❌ {dataset_name}: No data available")
        return
    
    sample = dataset[sample_index]
    
    print(f"\n🔍 {dataset_name}")
    print("─" * 60)
    
    # Print data source
    print(f"📁 Data Source: {sample.get('data_source', 'N/A')}")
    
    # Print abilities
    print(f"🎯 Abilities: {sample.get('abilities', 'N/A')}")
    
    # Print image info
    image_count = len(sample.get('images', []))
    print(f"🖼️  Images: {image_count}")
    
    # Print prompt
    prompt = sample.get('prompt', [])
    if prompt and len(prompt) > 0:
        user_content = prompt[0].get('content', '')
        # Truncate long content for display
        if len(user_content) > 200:
            user_content = user_content[:200] + "..."
        print(f"💬 User Prompt: {user_content}")
    
    # Print ground truth
    reward_model = sample.get('reward_model', {})
    ground_truth = reward_model.get('ground_truth', 'N/A')
    if len(str(ground_truth)) > 100:
        ground_truth = str(ground_truth)[:100] + "..."
    print(f"✅ Ground Truth: {ground_truth}")
    
    # Print extra info
    extra_info = sample.get('extra_info', {})
    if 'question' in extra_info:
        question = extra_info['question']
        if len(str(question)) > 150:
            question = str(question)[:150] + "..."
        print(f"❓ Question: {question}")
    
    print("─" * 60)

# Print sample from each dataset
for i, dataset in enumerate(sampled_datasets):
    if len(dataset) > 0:
        source_name = dataset[0]["data_source"]
        print_dataset_sample(dataset, source_name, sample_index=0)
    else:
        print(f"\n❌ Dataset {i+1}: Empty dataset")

# %% Ensure unique image_id across all datasets
print("\n" + "=" * 80)
print("🔐 ENSURING UNIQUE IMAGE_ID ACROSS ALL DATASETS")
print("=" * 80)

def ensure_unique_image_ids(dataset, prefix=""):
    """Ensure all image_ids in the dataset are unique by adding prefix if needed"""
    if len(dataset) == 0:
        return dataset
    
    # Check for duplicate image_ids
    all_image_ids = []
    for item in dataset:
        image_ids = item.get('image_id', [])
        if isinstance(image_ids, list):
            all_image_ids.extend(image_ids)
        else:
            all_image_ids.append(image_ids)
    
    # If no duplicates, return as is
    if len(all_image_ids) == len(set(all_image_ids)):
        print(f"✅ {prefix or 'Dataset'}: All image_ids are already unique")
        return dataset
    
    # If duplicates found, regenerate with prefix
    print(f"🔄 {prefix or 'Dataset'}: Regenerating image_ids to ensure uniqueness")
    
    def regenerate_image_ids(item):
        new_item = item.copy()
        old_image_ids = new_item.get('image_id', [])
        
        if isinstance(old_image_ids, list):
            new_image_ids = []
            new_sandbox_paths = []
            for i, old_id in enumerate(old_image_ids):
                new_id = f"{prefix}_{uuid.uuid4().hex}" if prefix else uuid.uuid4().hex
                new_image_ids.append(new_id)
                new_sandbox_paths.append(f"./{new_id}.jpg")
            
            new_item['image_id'] = new_image_ids
            new_item['sandbox_image_path'] = new_sandbox_paths
        else:
            new_id = f"{prefix}_{uuid.uuid4().hex}" if prefix else uuid.uuid4().hex
            new_item['image_id'] = [new_id]
            new_item['sandbox_image_path'] = [f"./{new_id}.jpg"]
        
        return new_item
    
    return dataset.map(regenerate_image_ids, num_proc=1)

# Ensure unique image_ids for each dataset
unique_datasets = []
for i, dataset in enumerate(sampled_datasets):
    if len(dataset) > 0:
        source_name = dataset[0]["data_source"]
        unique_dataset = ensure_unique_image_ids(dataset, prefix=source_name)
        unique_datasets.append(unique_dataset)
    else:
        unique_datasets.append(dataset)

# Final uniqueness check across all datasets
print("\n🔍 Final uniqueness check across all datasets...")
all_final_image_ids = []
for dataset in unique_datasets:
    for item in dataset:
        image_ids = item.get('image_id', [])
        if isinstance(image_ids, list):
            all_final_image_ids.extend(image_ids)
        else:
            all_final_image_ids.append(image_ids)

unique_count = len(set(all_final_image_ids))
total_count = len(all_final_image_ids)

if unique_count == total_count:
    print(f"✅ SUCCESS: All {total_count} image_ids are unique!")
else:
    print(f"❌ WARNING: Found {total_count - unique_count} duplicate image_ids")
    print(f"   Total: {total_count}, Unique: {unique_count}")

# Use unique datasets for final concatenation
final_dataset = concatenate_datasets(unique_datasets)

print(f"\n📊 Final dataset size: {len(final_dataset)} samples")
print(f"🔑 Total unique image_ids: {len(set(all_final_image_ids))}")

# Save to parquet file
output_path = f"/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_val/lmm_eval_lite_{len(final_dataset)//1000}k.parquet"
# print(f"\n💾 Saving final dataset to: {output_path}")
from datasets import Features, Sequence, Value

# final_dataset.to_parquet(output_path, index=False)
# final_dataset = final_dataset.cast(features)
print(final_dataset.features)
final_dataset.push_to_hub("ColeYzzzz/lmm_eval_lite", max_shard_size="500MB")
final_dataset.to_parquet(output_path, index=False)

print("=" * 80)
print("🎉 DATA PROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 80)
# print(f"📁 Output file: {output_path}")
print(f"📊 Total samples: {len(final_dataset)}")
print(f"🔑 Unique image_ids: {len(set(all_final_image_ids))}")
print("=" * 80)

# Check image count consistency
print("\n" + "=" * 80)
print("🔍 CHECKING IMAGE COUNT CONSISTENCY")
print("=" * 80)

def check_image_consistency(dataset, dataset_name):
    """Check if <image> count matches images count in each sample"""
    inconsistent_count = 0
    total_samples = len(dataset)
    
    for i, item in enumerate(dataset):
        prompt_content = item.get('prompt', [{}])[0].get('content', '')
        images = item.get('images', [])
        
        # Count <image> tags in prompt content
        image_tags_count = prompt_content.count('<image>')
        actual_images_count = len(images)
        
        if image_tags_count != actual_images_count:
            inconsistent_count += 1
            if inconsistent_count <= 5:  # Only show first 5 errors
                print(f"⚠️  {dataset_name} - Sample {i}:")
                print(f"   Prompt <image> count: {image_tags_count}")
                print(f"   Actual images count: {actual_images_count}")
                print(f"   Content preview: {prompt_content}...")
                print(f"   Question: {item.get('extra_info', {}).get('question', 'N/A')}")
                print(f"   Answer: {item.get('extra_info', {}).get('answer', 'N/A')}")
                print()
    
    if inconsistent_count == 0:
        print(f"✅ {dataset_name}: All {total_samples} samples have consistent image counts")
    else:
        print(f"❌ {dataset_name}: {inconsistent_count}/{total_samples} samples have inconsistent image counts")
    
    return inconsistent_count

# Check each dataset
total_inconsistent = 0
for dataset in unique_datasets:
    if len(dataset) > 0:
        source_name = dataset[0]["data_source"]
        inconsistent = check_image_consistency(dataset, source_name)
        total_inconsistent += inconsistent

print("=" * 80)
if total_inconsistent == 0:
    print("🎉 ALL DATASETS HAVE CONSISTENT IMAGE COUNTS!")
else:
    print(f"⚠️  TOTAL: {total_inconsistent} samples have inconsistent image counts")
print("=" * 80)

# Final dataset summary
print("\n📋 FINAL DATASET SUMMARY:")
print("─" * 50)
for i, dataset in enumerate(unique_datasets):
    if len(dataset) > 0:
        source_name = dataset[0]["data_source"]
        image_count = sum(len(item.get('image_id', [])) for item in dataset)
        print(f"{i+1:2d}. {source_name:<25} | {len(dataset):4d} samples | {image_count:4d} images")
    else:
        print(f"{i+1:2d}. {'Empty Dataset':<25} | {len(dataset):4d} samples | {0:4d} images")
