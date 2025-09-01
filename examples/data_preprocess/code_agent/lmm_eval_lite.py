import logging
import os

import datasets

# logging.basicConfig(level=logging.INFO)
# datasets.logging.set_verbosity_debug()
# os.environ["HF_HOME"] = "/tmp/hf"
# os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_ds"
# from datasets import load_dataset, DownloadConfig
# ds = load_dataset(
#     "ColeYzzzz/lmm_eval_lite",
#     split="train",
# )
# print(ds)
# ds = "/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_val/lmm_eval_lite.parquet"
# ds = datasets.load_dataset("parquet", data_files=ds)["train"]
# # ds.to_parquet(
# #     "/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_val/lmm_eval_lite.parquet"
# # )
# # 随机选100条
# import random
# ds.select(random.sample(range(len(ds)), 100)).to_parquet(
#     "/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_val/lmm_eval_lite_100.parquet"
# )
# ds = "/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_train/deepeyes_train_47k.parquet"
# ds = datasets.load_dataset("parquet", data_files=ds)["train"]
# # 随机选100条
# import random
# ds.select(random.sample(range(len(ds)), 100)).to_parquet(
#     "/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_train/deepeyes_train_100_debug.parquet"
# )
from datasets import load_dataset

ds = datasets.load_dataset("parquet", data_files="/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/o3_rl_val/lmm_eval_lite_v2.parquet", split="train")

print(ds[0])