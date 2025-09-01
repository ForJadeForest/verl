import os
import random

from datasets import load_dataset

eval_path = "/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_val/lmm_eval_lite_v2.parquet"


def filter_item(item):
    if item["data_source"] in ["HallusionBench", "VStar", "TreeBench", "MMStar"]:
        return True
    return False

def get_vstar_ds(item):
    if item["data_source"] == "VStar":
        return True
    return False

def get_hallusion_bench_ds(item):
    if item["data_source"] == "HallusionBench":
        return True
    return False

def get_tree_bench_ds(item):
    if item["data_source"] == "TreeBench":
        return True
    return False

def get_mmstar_ds(item):
    if item["data_source"] == "MMStar":
        return True
    return False


if __name__ == "__main__":
    ds = load_dataset("parquet", data_files=eval_path)["train"]
    print(ds)

    ds = ds.filter(filter_item, num_proc=16)
    print(ds)
    vstar_ds = ds.filter(get_vstar_ds, num_proc=16)
    hallusion_bench_ds = ds.filter(get_hallusion_bench_ds, num_proc=16)
    hallusion_bench_ds = hallusion_bench_ds.select(random.sample(range(len(hallusion_bench_ds)), 200))
    tree_bench_ds = ds.filter(get_tree_bench_ds, num_proc=16)
    mmstar_ds = ds.filter(get_mmstar_ds, num_proc=16)
    mmstar_ds = mmstar_ds.select(random.sample(range(len(mmstar_ds)), 200))

    print(vstar_ds)
    print(hallusion_bench_ds)
    print(tree_bench_ds)
    print(mmstar_ds)
    from datasets import concatenate_datasets

    val_ds = concatenate_datasets([vstar_ds, hallusion_bench_ds, tree_bench_ds, mmstar_ds])

    save_path = "/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_val/lmm_eval_lite_perception_1k.parquet"
    val_ds.to_parquet(save_path)
    print(val_ds)

    print(f"save to {save_path}")