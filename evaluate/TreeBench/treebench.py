import argparse
import asyncio
import base64
import json
import os
import sys
import time
import uuid
from io import BytesIO

from datasets import load_dataset

# OpenAI 兼容客户端（优先异步）
from openai import AsyncOpenAI as OpenAIClient
from PIL import Image
from tqdm import tqdm

from evaluate.infer_engine_utils import (
    build_user_message_for_gen,
    extract_answer,
    extract_response,
    solve_one_query,  # code tool 推理引擎（多轮+沙箱）
)
from evaluate.prompt import get_system_prompt, query_template
from evaluate.utils import encode_image_base64, qwen_resize_image

# --------------------------
# 参数解析
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
parser.add_argument("--eval_api_url", type=str, default="http://29.157.70.224:8000/v1", help="评测模型 API URL")
parser.add_argument("--judge_api_url", type=str, default="http://28.12.131.135:8000/v1", help="判题模型 API URL")
parser.add_argument("--save_path", type=str, default="./results/evaluation/treebench", help="结果保存根路径")
parser.add_argument("--use_code_tool", action="store_true", help="启用 code tool 评测")
parser.add_argument("--sandbox_url", type=str, default="http://29.157.70.224:8080", help="代码沙箱 URL")
parser.add_argument("--num_workers", type=int, default=12, help="并发度（异步信号量/任务数）")
parser.add_argument("--pre_resize", action="store_true", help="预先resize图片")
args = parser.parse_args()
time_str = time.strftime("%Y%m%d_%H%M")

# 判题 few-shot
def get_chat_template() -> str:
    return (
        "Below are two answers to a question. Question is [Question], "
        "[Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from "
        "a model's output to this question.  Determine whether these two answers are consistent.\n"
        "Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. "
        "If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.\n"
        "If they are consistent, Judement is 1; if they are different, Judement is 0. "
        "Just output Judement and don't output anything else.\n\n"
    )


def get_gpt4_score_ICE() -> list[str]:
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: A. The countertop is tan.
[Model_answer] : tan
Judgement: 1
"""
    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: A. The barrier is on the left side of the picture.
[Model_answer] : A
Judgement: 1
"""
    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: A. Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
"""
    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: A. No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
"""
    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: A. The boy is wearing pants.
[Model_answer] : C. The girl in the picture is wearing pants.
Judgement: 0
"""
    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: A. Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
"""
    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: A. The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
"""
    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]

def build_judge_prompt(predict_str: str, standard_answer: str, question: str) -> str:
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template + "\n\n".join(examples) + "\n\n"
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {standard_answer}
[Model_answer] : {predict_str}
Judgement:"""
    return f"{demo_prompt}{test_prompt}"

def build_client(api_url: str):
    return OpenAIClient(
        api_key=args.api_key,
        base_url=api_url,
    )


def build_question_prompt(item) -> str:
    # TreeBench: multi-choice in a single string field
    # question + "\n" + options block
    qs = item["question"]
    if "multi-choice options" in item and item["multi-choice options"]:
        qs = qs + "\n" + item["multi-choice options"]
    return qs

async def process_one_item(
    item: dict,
    eval_client,
    judge_client,
    eval_model_name: str,
    judge_model_name: str,
    semaphore: asyncio.Semaphore,
    sandbox_url: str,
):
    """
    对单条进行推理与判题（LLM-as-judge）。
    返回：save_info(dict), acc_int(0/1)
    """
    # Build question text
    question = build_question_prompt(item)

    # Image handling
    base64_image = item.get("image")
    if base64_image is None:
        raise ValueError("Item missing 'image' field")

    # Optional resize
    if args.pre_resize:
        try:
            pil_img = Image.open(BytesIO(base64.b64decode(base64_image)))
        except Exception:
            # If it's not base64 string, try to treat as PIL Image
            pil_img = item.get("image_pil")
            if pil_img is None:
                raise
        pil_img = qwen_resize_image(
            pil_img,
            max_pixels=8192 * 28 * 28 * 2,
        )
        base64_image = encode_image_base64(pil_img)

    image_uuid = uuid.uuid4().hex
    upload_img_paths = [f"./{image_uuid}.jpg"]

    # Prompting
    if args.use_code_tool:
        system_message = get_system_prompt()
        user_text = query_template(question, upload_img_paths)
        user_content = build_user_message_for_gen(base64_image, user_text)
        upload_image_dict = {upload_img_paths[0]: base64_image}
    else:
        system_message = "You are a helpful assistant."
        user_text = "<image>" + question
        user_content = build_user_message_for_gen(base64_image, user_text)
        upload_image_dict = None

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]

    # Generation
    gen_kwargs = {
        "temperature": 0.0,
        "max_tokens": 10240,
        "top_p": 1.0,
        # "extra_body": {
        #     "repetition_penalty": 1.05 if args.use_code_tool else 1.0,
        # },
    }

    output_messages = []
    status = "Success"
    async with semaphore:
        if args.use_code_tool:
            try:
                output_messages = await solve_one_query(
                    messages,
                    upload_image_dict,
                    eval_client,
                    eval_model_name,
                    sandbox_url,
                    max_turn=10,
                    gen_kwargs=gen_kwargs,
                )
                response = extract_response(output_messages)
            except Exception as e:
                response = f"<error>{e}</error>"
                status = f"GenError: {e}"
                output_messages = messages
        else:
            try:
                resp = await eval_client.chat.completions.create(
                    model=eval_model_name,
                    messages=messages,
                    **gen_kwargs,
                )
                response = resp.choices[0].message.content
                output_messages = messages + [{"role": "assistant", "content": response}]
            except Exception as e:
                response = f"<error>{e}</error>"
                status = f"GenError: {e}"
                output_messages = messages

    pred_output = extract_answer(response)

    # LLM-as-judge (TreeBench answer is single option like 'A'/'B'/...)
    standard_answer = item.get("answer")
    if standard_answer is None:
        raise ValueError("Item missing 'answer' field")

    judge_prompt = (
        "You will compare a model's answer with the ground-truth option for a multiple-choice question.\n"
        "If they are essentially the same (e.g., 'B' vs 'The answer is B'), output 1; otherwise 0.\n\n"
        f"[Question]: {question}\n"
        f"[Standard Answer]: {standard_answer}\n"
        f"[Model_answer] : {pred_output}\n"
        "Judgement:"
    )

    try:
        jresp = await judge_client.chat.completions.create(
            model=judge_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0.0,
        )
        jtxt = jresp.choices[0].message.content.strip()
        if "Judgement:" in jtxt:
            jtxt = jtxt.split("Judgement:")[-1].strip()
        acc_reward = 1.0 if "1" in jtxt else 0.0
    except Exception as e:
        print(f"[FATAL] 判题失败。错误：{e}")
        acc_reward = 0.0
        status = f"{status} | JudgeError: {e}"

    save_info = {
        "question": question,
        "answer": standard_answer,
        "pred_output": pred_output,
        "response": response,
        "messages": output_messages,
        "status": status,
        "acc": float(acc_reward),
        "user_text": user_text,
        "prediction": pred_output,
        "category": item.get("category", "Unknown"),
    }
    return save_info, int(acc_reward)

async def main():
    # 创建两个客户端
    eval_client = build_client(args.eval_api_url)
    judge_client = build_client(args.judge_api_url)

    # 从各自的 /models 获取模型列表
    try:
        eval_resp = await eval_client.models.list()
        eval_model_name = eval_resp.data[0].id
    except Exception as e:
        print(f"[FATAL] 无法从评测API获取模型列表。错误：{e}")
        sys.exit(1)

    try:
        judge_resp = await judge_client.models.list()
        judge_model_name = judge_resp.data[0].id
    except Exception as e:
        print(f"[FATAL] 无法从判题API获取模型列表。错误：{e}")
        sys.exit(1)

    print(f"Using eval model: {eval_model_name} from {args.eval_api_url}")
    print(f"Using judge model: {judge_model_name} from {args.judge_api_url}")

    # 加载 TreeBench 数据集
    ds = load_dataset("HaochenWang/TreeBench", split="train")

    # 输出目录
    model_name_safe = eval_model_name.replace("/", "_")
    judge_model_name_safe = judge_model_name.replace("/", "_")
    code_tool_suffix = "use_code" if args.use_code_tool else "no_code"
    save_dir = os.path.join(
        args.save_path,
        f"{model_name_safe}_{code_tool_suffix}_judge-{judge_model_name_safe}/{time_str}",
    )
    os.makedirs(save_dir, exist_ok=True)

    result_jsonl = os.path.join(save_dir, "result.jsonl")
    raw_msg_jsonl = os.path.join(save_dir, "result_raw_messages.jsonl")

    semaphore = asyncio.Semaphore(args.num_workers)
    pbar = tqdm(total=len(ds), desc="TreeBench", dynamic_ncols=True)

    correct_count = 0
    processed = 0

    results_to_write = []
    raw_messages_to_write = []

    async def _task(item):
        return await process_one_item(
            item=item,
            eval_client=eval_client,
            judge_client=judge_client,
            eval_model_name=eval_model_name,
            judge_model_name=judge_model_name,
            semaphore=semaphore,
            sandbox_url=args.sandbox_url,
        )

    # 分批提交任务，避免一次性创建全部协程导致内存和调度压力
    total = len(ds)
    for start in range(0, total, args.num_workers):
        batch_indices = range(start, min(start + args.num_workers, total))
        tasks = [asyncio.create_task(_task(ds[i])) for i in batch_indices]
        for coro in asyncio.as_completed(tasks):
            res, acc_int = await coro
            # Separate raw messages for compatibility
            raw_messages_to_write.append(res.get("messages", []))
            res.pop("messages", None)
            results_to_write.append(res)
            processed += 1
            correct_count += acc_int
            cur_acc = (correct_count / processed) * 100.0 if processed else 0.0
            pbar.set_postfix(accuracy=f"{cur_acc:.2f}%")
            pbar.update(1)

    pbar.close()

    with open(raw_msg_jsonl, "w", encoding="utf-8") as f:
        for item in raw_messages_to_write:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(result_jsonl, "w", encoding="utf-8") as f:
        for item in results_to_write:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 指标计算参考提供的代码片段（按类别 + overall + mean IoU）
    tags = [
        "Perception/Attributes",
        "Perception/Material",
        "Perception/Physical State",
        "Perception/Object Retrieval",
        "Perception/OCR",
        "Reasoning/Perspective Transform",
        "Reasoning/Ordering",
        "Reasoning/Contact and Occlusion",
        "Reasoning/Spatial Containment",
        "Reasoning/Comparison",
    ]
    results = {t: {"correct": 0, "total": 0} for t in tags}
    total = 0
    correct = 0
    for it in results_to_write:
        tag = it.get("category")
        if tag in results:
            results[tag]["total"] += 1
            total += 1
            if int(it.get("acc", 0)) == 1:
                results[tag]["correct"] += 1
                correct += 1

    # 打印各类精度
    for tag in tags:
        t_total = results[tag]["total"]
        t_correct = results[tag]["correct"]
        acc = (t_correct / t_total) if t_total else 0.0
        print(f"{tag} {t_correct}/{t_total}={round(acc * 100, 2)}")

    overall_acc = (correct / total * 100.0) if total else 0.0
    print(f"==> Overall {correct}/{total}={round(overall_acc, 2)}")

    # 生成最终汇总 JSON
    overall_acc = (correct_count / processed * 100.0) if processed else 0.0
    final_json_path = os.path.join(save_dir, "final_acc.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        summary = {
            "overall": overall_acc,
            "count": processed,
            "correct": correct_count,
            "error_nums": processed - correct_count,
            "per_category": {
                tag: {
                    "correct": results[tag]["correct"],
                    "total": results[tag]["total"],
                    "acc": (results[tag]["correct"] / results[tag]["total"] * 100.0) if results[tag]["total"] else 0.0,
                }
                for tag in tags
            },
        }
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print(f"Saved results to: {save_dir}")


if __name__ == "__main__":
    asyncio.run(main())
