import argparse
import asyncio
import base64
import json
import os
import sys
import time
import uuid
from io import BytesIO

import pandas as pd
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
parser.add_argument("--eval_api_url", type=str, default="http://28.12.130.184:8000/v1", help="评测模型 API URL")
parser.add_argument("--judge_api_url", type=str, default="http://28.12.131.135:8000/v1", help="判题模型 API URL")
parser.add_argument("--hrbench_path", type=str, default="/root/data/HR-Bench", help="HRBench 数据集路径")
parser.add_argument("--save_path", type=str, default="./results/evaluation/hrbench", help="结果保存根路径")
parser.add_argument("--use_code_tool", action="store_true", help="启用 code tool 评测")
parser.add_argument("--sandbox_url", type=str, default="http://28.12.130.184:8080", help="代码沙箱 URL")
parser.add_argument("--num_workers", type=int, default=12, help="并发度（异步信号量/任务数）")
parser.add_argument("--pre_resize", action="store_true", help="预先resize图片")
args = parser.parse_args()
time_str = time.strftime("%Y%m%d_%H%M")


# --------------------------
# 常量与模板
# --------------------------
TEST_TYPES = ["hr_bench_4k", "hr_bench_8k"]
ABC_MAP = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"}

INSTRUCTION_PROMPT_BEFORE = """Question: {question}
Options: {options}""".strip()


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


# --------------------------
# OpenAI 兼容客户端（异步优先，回落同步）
# --------------------------
def build_client(api_url: str):
    return OpenAIClient(
        api_key=args.api_key,
        base_url=api_url,
    )


def rule_judge(pred_ans: str, standard_answer: str) -> float:
    """
    规则判定（与HRBench原判题逻辑保持一致）：
    - 如果仅单字符且为标准答案 => 正确
    - 如果是两字符且有 '.' 且含标准答案 => 正确
    - 如果标准答案出现在模型答案中 => 正确
    否则未知，返回 -1 表示需转LLM判定
    """
    clean = pred_ans.strip()

    if "\\boxed" in clean:
        try:
            clean = clean.split("\\boxed{")[1].split("}")[0]
        except Exception:
            pass

    # 单字符
    if len(clean) == 1:
        return 1.0 if clean == standard_answer else 0.0
    # 两字符且有 '.'
    if len(clean) == 2 and "." in clean:
        return 1.0 if standard_answer in clean else 0.0
    # 标准答案片段出现
    # if standard_answer in clean:
    #     return 1.0
    # 未能确定，交给LLM判定
    return -1.0


async def process_one_item(
    eval_client,
    judge_client,
    eval_model_name: str,
    judge_model_name: str,
    df_row: pd.Series,
    use_code_tool: bool,
    semaphore: asyncio.Semaphore,
    sandbox_url: str,
):
    """
    返回：
      result_dict, per_item_acc(0/1), test_type
    """
    # 从DataFrame行获取数据
    img_base64 = df_row["image"]
    question = df_row["question"]
    answer = df_row["answer"]
    answer_str = df_row[answer]
    category = df_row["category"]

    # 处理图片
    if args.pre_resize:
        pil_img = qwen_resize_image(
            Image.open(BytesIO(base64.b64decode(img_base64))),
            max_pixels=8192 * 28 * 28 * 2,
        )
        img_base64 = encode_image_base64(pil_img)

    base64_image = img_base64
    # 选项字符串
    options = [
        "A. " + df_row["A"],
        "B. " + df_row["B"],
        "C. " + df_row["C"],
        "D. " + df_row["D"],
    ]
    option_str = "\n" + "\n".join(options)

    # prompt 组装
    if use_code_tool:
        system_message = get_system_prompt()
        image_uuid = str(uuid.uuid4())
        upload_img_paths = f"./{image_uuid}.jpg"
        question = INSTRUCTION_PROMPT_BEFORE.format(question=question, options=option_str)
        user_text = query_template(question, upload_img_paths)
        upload_image_dict = {upload_img_paths: base64_image}
        user_content = build_user_message_for_gen(base64_image, user_text)
    else:
        system_message = "You are a helpful assistant."
        question = "<image>" + question
        user_text = INSTRUCTION_PROMPT_BEFORE.format(question=question, options=option_str)
        user_content = build_user_message_for_gen(base64_image, user_text)
        upload_image_dict = None  # 不使用 code tool 时无需这个

    # ========== 生成 ==========
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]

    pred_output = ""
    output_messages = []
    status = "Success"

    # gen kwargs
    gen_kwargs = {
        "temperature": 0.0,
        "max_tokens": 10240,
        "top_p": 1.0,
        # "extra_body": {
        #     "repetition_penalty": 1.05 if use_code_tool else 1.0,
        # },
    }

    async with semaphore:
        if use_code_tool:
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
        else:
            resp = await eval_client.chat.completions.create(
                model=eval_model_name,
                messages=messages,
                **gen_kwargs,
            )
            response = resp.choices[0].message.content
            output_messages.append({"role": "assistant", "content": response})
    pred_output = extract_answer(response)

    # ========== 判题（先规则，必要时LLM） ==========
    acc_reward = rule_judge(pred_output, answer)

    if acc_reward < 0:  # 交给 LLM 判定
        full_prompt = build_judge_prompt(pred_output, f"{answer}. {answer_str}", question)
        judge_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt},
        ]
        try:
            jresp = await judge_client.chat.completions.create(
                model=judge_model_name,
                messages=judge_messages,
                temperature=0.0,
            )
            jtxt = jresp.choices[0].message.content.strip()
            # 兼容两种输出风格
            if "Judgement:" in jtxt:
                jtxt = jtxt.split("Judgement:")[-1].strip()
            acc_reward = 1.0 if "1" in jtxt else 0.0
        except Exception as e:
            # 判题失败按错误处理为错误（保守）
            acc_reward = 0.0
            status = f"{status} | JudgeError: {e}"

    # 保存单条
    save_info = {
        "question": question,
        "answer": answer,
        "answer_str": answer_str,
        "pred_output": pred_output,
        "response": response,
        "messages": output_messages,
        "status": status,
        "category": category,
        "acc": float(acc_reward),
        "user_text": user_text,
    }
    return save_info, int(acc_reward)


# --------------------------
# 每个类型目录的批处理（并发 + 实时准确率）
# --------------------------
async def process_one_type(
    eval_client,
    judge_client,
    eval_model_name: str,
    judge_model_name: str,
    type_name: str,
    hrbench_path: str,
    save_root: str,
    use_code_tool: bool,
    max_concurrency: int,
    sandbox_url: str,
):
    tsv_path = os.path.join(hrbench_path, type_name + ".tsv")
    df = pd.read_csv(tsv_path, sep="\t")
    total = df.shape[0]

    # 输出目录
    code_tool_suffix = "use_code" if use_code_tool else "no_code"
    model_name_safe = eval_model_name.replace("/", "_")
    judge_model_name_safe = judge_model_name.replace("/", "_")
    
    save_dir = os.path.join(
        save_root,
        f"{model_name_safe}_{code_tool_suffix}_judge-{judge_model_name_safe}",
        time_str,
    )

    # 与原脚本兼容的两个文件
    result_jsonl = os.path.join(save_dir, f"result_{type_name}.jsonl")
    raw_msg_jsonl = os.path.join(save_dir, f"result_{type_name}_raw_messages.jsonl")

    semaphore = asyncio.Semaphore(max_concurrency)

    # 进度条 + 实时准确率
    correct_count = 0
    processed = 0

    results_to_write = []
    raw_messages_to_write = []

    # tqdm 配置
    pbar = tqdm(total=total, desc=f"HRBench {type_name}", dynamic_ncols=True)

    async def _task(idx: int):
        return await process_one_item(
            eval_client=eval_client,
            judge_client=judge_client,
            eval_model_name=eval_model_name,
            judge_model_name=judge_model_name,
            df_row=df.iloc[idx],
            use_code_tool=use_code_tool,
            semaphore=semaphore,
            sandbox_url=sandbox_url,
        )

    # 分批提交任务，避免一次性创建全部协程导致内存和调度压力
    for start in range(0, total, max_concurrency):
        batch_idx = list(range(start, min(start + max_concurrency, total)))
        tasks = [asyncio.create_task(_task(idx)) for idx in batch_idx]
        for coro in asyncio.as_completed(tasks):
            res, acc_int = await coro
            results_to_write.append(res)
            raw_messages_to_write.append(res.pop("messages", []))  # 与原一致：消息独立保存
            processed += 1
            correct_count += acc_int
            cur_acc = (correct_count / processed) * 100.0 if processed else 0.0
            pbar.set_postfix(accuracy=f"{cur_acc:.2f}%")
            pbar.update(1)

    pbar.close()
    os.makedirs(save_dir, exist_ok=True)
    # 写盘（逐行 JSON）
    with open(raw_msg_jsonl, "w", encoding="utf-8") as f:
        for item in raw_messages_to_write:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(result_jsonl, "w", encoding="utf-8") as f:
        for item in results_to_write:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 返回统计
    per_type_acc = correct_count / total if total else 0.0
    return {
        "type": type_name,
        "save_dir": save_dir,
        "result_jsonl": result_jsonl,
        "raw_msg_jsonl": raw_msg_jsonl,
        "acc": per_type_acc,
        "count": total,
        "correct": correct_count,
        "items": results_to_write,
    }


# --------------------------
# 主流程
# --------------------------
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

    # 汇总路径根
    save_root = args.save_path

    # 分类型并发执行（也可串行；这里串行以便输出结构与目录更清晰）
    per_type_stats = []
    overall_correct = 0
    overall_count = 0

    for tname in TEST_TYPES:
        stats = await process_one_type(
            eval_client=eval_client,
            judge_client=judge_client,
            eval_model_name=eval_model_name,
            judge_model_name=judge_model_name,
            type_name=tname,
            hrbench_path=args.hrbench_path,
            save_root=save_root,
            use_code_tool=args.use_code_tool,
            max_concurrency=args.num_workers,
            sandbox_url=args.sandbox_url,
        )
        per_type_stats.append(stats)
        overall_correct += stats["correct"]
        overall_count += stats["count"]

    overall_acc = overall_correct / overall_count if overall_count else 0.0

    # 汇总 final_acc.json（放在最后一个类型目录下，或单独顶层）
    final_out_dir = (
        per_type_stats[-1]["save_dir"]
        if per_type_stats
        else os.path.join(
            save_root,
            f"{eval_model_name.replace('/', '_')}_{'use_code' if args.use_code_tool else 'no_code'}_judge-{judge_model_name.replace('/', '_')}_{time_str}",
        )
    )
    os.makedirs(final_out_dir, exist_ok=True)

    final_json_path = os.path.join(final_out_dir, "final_acc.json")

    final_acc = {t["type"]: t["acc"] * 100.0 for t in per_type_stats}
    final_acc["overall"] = overall_acc * 100.0

    # 与原脚本的 error_nums_type / error_preds 类似产物
    error_nums_type = {t["type"]: int(t["count"] - t["correct"]) for t in per_type_stats}
    final_acc["error_nums"] = error_nums_type

    # 汇总错误样例（跨类型）
    error_preds = []
    for t in per_type_stats:
        for item in t["items"]:
            if int(item.get("acc", 0)) != 1:
                error_preds.append(
                    {
                        "pred_ans": item.get("pred_output", ""),
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                    }
                )
    final_acc["error_preds"] = error_preds

    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_acc, f, ensure_ascii=False, indent=4)

    # 控制台打印最终结果
    for t in per_type_stats:
        print(f"Accuracy for {t['type']}: {t['acc'] * 100.0:.2f}%  (correct {t['correct']}/{t['count']})")
    print(f"Overall Accuracy: {overall_acc * 100.0:.2f}%  (correct {overall_correct}/{overall_count})")
    print(f"Saved final summary to: {final_json_path}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user")
