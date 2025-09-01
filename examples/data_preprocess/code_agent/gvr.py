from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset
from PIL import Image

# 假设你已将 transform_cot 定义在同一模块或已导入
# from your_module import transform_cot

# 1) 读数据
data_file = "/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/VGR/data/vgr_longcot.parquet"
image_dir = "/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/LLaVA-NeXT-Data/llava_next_raw_format_images"
ds = load_dataset("parquet", data_files=data_file, split="train")
print(ds[0])


# 2) 示例：添加 source 列（你已有）
def get_source(item: dict[str, Any]) -> dict[str, Any]:
    source = item["image"].split("/")[0]
    return {"source": source}


ds = ds.map(get_source, num_proc=10)
print(set(ds["source"]))


# 3) 主转换：生成 final_cot（以及可选的裁剪信息）
BASE_CROP_DIR = Path("/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/VGR/data/image_crops")
BASE_CROP_DIR.mkdir(parents=True, exist_ok=True)



_SOT_EOT_RE = re.compile(r"<SOT>\s*\[([^\]]+)\]\s*<EOT>")
_FINAL_ANS_RE = re.compile(r"Final answer\s*:\s*(.*)\s*\Z", re.DOTALL)


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


def _parse_coord_block(block: str) -> tuple[float, float, float, float]:
    """
    将 "[0.1, 0.02, 0.8, 0.98]" 风格的内容解析为 4 个 float。
    """
    parts = [p.strip() for p in block.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox 元素数量不是 4")
    try:
        x1, y1, x2, y2 = (float(p) for p in parts)
    except ValueError as exc:  # noqa: TRY003
        raise ValueError("bbox 解析失败（非数字）") from exc
    return x1, y1, x2, y2


def _to_abs_bbox(rel: tuple[float, float, float, float], w: int, h: int) -> BBox:
    """
    将相对坐标 (x1,y1,x2,y2) ∈ [0,1] 映射为像素绝对坐标（含边界裁剪）。
    """
    rx1, ry1, rx2, ry2 = rel

    # 基本合法性检查（顺序 & 范围主体落在 0..1；允许极小越界后续再 clamp）
    if rx2 <= rx1 or ry2 <= ry1:
        raise ValueError("bbox 顺序非法：x2<=x1 或 y2<=y1")
    ax1 = max(0, min(int(round(rx1 * w)), w - 1))
    ay1 = max(0, min(int(round(ry1 * h)), h - 1))
    ax2 = max(0, min(int(round(rx2 * w)), w))
    ay2 = max(0, min(int(round(ry2 * h)), h))

    if ax2 - ax1 <= 1 or ay2 - ay1 <= 1:
        raise ValueError("bbox 面积过小或无效")

    return BBox(ax1, ay1, ax2, ay2)


def _iter_sot_eot(text: str) -> Iterator[re.Match[str]]:
    """
    迭代所有 <SOT>...[...]<EOT> 的匹配。
    """
    return _SOT_EOT_RE.finditer(text)


def _replace_boxes_with_abs(text: str, abs_bboxes: list[BBox]) -> str:
    """
    将按出现顺序的 <SOT>[...rel...]<EOT> 逐一替换为 <box>[abs]</box>。
    """
    out: list[str] = []
    last_end = 0
    for idx, m in enumerate(_iter_sot_eot(text)):
        out.append(text[last_end : m.start()])
        if idx >= len(abs_bboxes):
            # 理论上不该发生：bbox 数量不匹配
            print("warning: 绝对坐标数量少于 <SOT>...<EOT> 出现次数；丢弃样本")
            return ""
        bb = abs_bboxes[idx]
        out.append(f"<box>[{bb.x1}, {bb.y1}, {bb.x2}, {bb.y2}]</box>")
        last_end = m.end()
    out.append(text[last_end:])
    return "".join(out)


# def _normalize_think_answer(text: str) -> str | None:
#     """
#     - 只保留开头第一个 <think>；去掉其余所有 <think>/<\\think>。
#     - 将 “\\n\\nFinal answer: xxx” 变为：
#         “\\n</think>\\n<answer> xxx </answer>”
#     - 若缺少 Final answer 或 <think> 不在开头，则告警并丢弃。
#     """
#     if not text.lstrip().startswith("<think>"):
#         print("warning: <think> 不在开头；丢弃样本")
#         return None

#     m = _FINAL_ANS_RE.search(text)
#     if not m:
#         print("warning: 未找到 'Final answer:'；丢弃样本")
#         return None

#     answer = m.group(1).strip()

#     # 去除除开头第一个以外的所有 <think>/<\\think>
#     # 思路：保留第一个 "<think>" 原样；把其余标签全部删掉
#     # 1) 定位首个 "<think>"
#     first_think_pos = text.find("<think>")
#     head = text[: first_think_pos + len("<think>")]
#     body = text[first_think_pos + len("<think>") :]

#     body = body.replace("<think>", "")
#     body = body.replace("</think>", "")

#     # 2) 替换 Final answer 段
#     body = _FINAL_ANS_RE.sub("\n</think>\n<answer> \\1 </answer>", body)

#     return head + body


def _replace_subsequent_image_tokens(text: str) -> str:
    """
    对除了第一个 <image> 的出现，替换为 '\\n\\n<SUBIMAGE>'.
    """
    parts = text.split("<image>")
    if len(parts) <= 2:
        return text
    # 重新拼接：保留第一次出现的 <image>，其余用 \n\n<SUBIMAGE> 代替
    rebuilt = parts[0] + "<image>"
    tail = ("\n\n<SUBIMAGE>").join(parts[1:])
    return rebuilt + tail


def transform_cot(item: dict[str, Any]) -> dict[str, Any] | None:
    """
    处理对话 COT：
    1) 提取 <SOT>...[rel]...<EOT>，转绝对坐标并裁剪图像，保存到本地；
    2) 将 COT 中的相对坐标替换为 <box>[abs]</box>；
    3) 除第一次 <image> 外，其余替换为 '\\n\\n<SUBIMAGE>'；
    4) 规范化 <think>/<answer>；若结构异常则丢弃样本（返回 None）。
    """
    if "conversations" not in item or len(item["conversations"]) < 2:
        print("warning: conversations 结构异常；丢弃样本")
        return None

    text: str = item["conversations"][1]["value"]
    image_path = Path(item.get("image", ""))
    image_path = image_dir / image_path

    if not image_path.is_file():
        print("warning: image_path 缺失或文件不存在；丢弃样本")
        return None

    # 读取原图尺寸
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            # 解析相对 bbox
            rel_list: list[tuple[float, float, float, float]] = []
            for m in _iter_sot_eot(text):
                try:
                    rel = _parse_coord_block(m.group(1))
                except ValueError as exc:
                    print(f"warning: 解析坐标失败：{exc}; 丢弃样本")
                    return None
                # 粗检：看起来像相对坐标（允许轻微越界，稍后 clamp）
                if any(not isinstance(v, float) for v in rel):
                    print("warning: 坐标并非浮点；丢弃样本")
                    return None
                rel_list.append(rel)

            # 相对 => 绝对；并完成裁剪保存
            abs_bboxes: list[BBox] = []
            cropped_info: list[dict[str, Any]] = []
            for i, rel in enumerate(rel_list):
                try:
                    bb = _to_abs_bbox(rel, w, h)
                except ValueError as exc:
                    print(f"warning: 坐标非法：{exc}; 丢弃样本")
                    return None
                abs_bboxes.append(bb)

                crop = im.crop((bb.x1, bb.y1, bb.x2, bb.y2))
                out_path = BASE_CROP_DIR / f"{image_path.stem}_crop_{i:02d}.jpg"
                crop.save(out_path, format="JPEG", quality=95)
                cropped_info.append(
                    {
                        "bbox_2d": bb.as_list(),
                        "cropped_image_path": str(out_path),
                    }
                )

    except Exception as exc:  # noqa: BLE001
        print(f"warning: 图像处理失败：{exc}; 丢弃样本")
        return None

    # 文本替换：<SOT>...[rel]...<EOT> -> <box>[abs]</box>
    new_text = _replace_boxes_with_abs(text, abs_bboxes)
    if not new_text:
        return None

    # 替换除第一个 <image> 之外的出现
    # new_text = _replace_subsequent_image_tokens(new_text)
    new_text = new_text.replace("<image>", "\n\n<SUBIMAGE>")

    # 规范化 <think>/<answer>
    # new_text = _normalize_think_answer(new_text)
    if new_text is None:
        return None
    new_text = new_text.replace("<think>", "").replace("</think>", "")
    item['final_cot'] = new_text
    return item


DEBUG = False
if DEBUG:
    ds = ds.select(range(10))

ds = ds.map(transform_cot, num_proc=16, desc="transforming COT -> final_cot")


# 4) 过滤掉失败样本（final_cot 为 None 的）
def keep_success(x: dict[str, Any]) -> bool:
    return x["final_cot"] is not None


success_ds = ds.filter(keep_success, num_proc=16)
for i in range(3):
    print(success_ds[0]['final_cot'])
    print("="*100)

# 5) 检查一下
print(f"Filter {len(ds)} -> {len(success_ds)}")

success_ds.to_parquet("/apdcephfs_gy5/share_303588738/yingzhepeng/datasets/VGR/data/vgr_longcot_final_cot.parquet")
