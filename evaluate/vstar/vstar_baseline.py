import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
from io import BytesIO

from openai import AsyncOpenAI as OpenAIClient
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
parser.add_argument("--eval_api_url", type=str, default="http://29.157.70.224:8000/v1", help="评测模型 API URL")
parser.add_argument("--judge_api_url", type=str, default="http://28.12.131.135:8000/v1", help="判题模型 API URL")
parser.add_argument("--vstar_bench_path", type=str, default="./data/vstar_bench", help="V* 数据集路径")
parser.add_argument("--save_path", type=str, default="./results/evaluation/vstar", help="结果保存根路径")
parser.add_argument("--num_workers", type=int, default=12, help="并发度（异步信号量/任务数）")
args = parser.parse_args()
time_str = time.strftime("%Y%m%d_%H%M")


TEST_TYPES = ["direct_attributes", "relative_position"]
ABC_MAP = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"}

INSTRUCTION_PROMPT_BEFORE = """Question: {question}
Options: {options}""".strip()


def extract_answer(response: str) -> str:
    search_result = re.search(r"<answer>(.*)</answer>", response, re.DOTALL)
    if search_result:
        response = search_result.group(1)
    else:
        response = response.strip()
    return response


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


def rule_judge(pred_ans: str, standard_answer_with_prefix: str) -> float:
    """
    规则判定（与你原判题逻辑保持一致）：
    - 如果仅单字符且为 'A' => 正确
    - 如果是 'A.' or 包含点且含 'A' => 正确
    - 如果标准答案子串出现在模型答案中 => 正确
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
        return 1.0 if clean == "A" else 0.0
    # 两字符且有 '.'
    if len(clean) == 2 and "." in clean:
        return 1.0 if "A" in clean else 0.0
    # 标准答案片段出现
    if standard_answer_with_prefix in clean:
        return 1.0
    # 未能确定，交给LLM判定
    return -1.0


def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


async def process_one_item(
    eval_client,
    judge_client,
    eval_model_name: str,
    judge_model_name: str,
    test_path: str,
    img_name: str,
    semaphore: asyncio.Semaphore,
):
    """
    返回：
      result_dict, per_item_acc(0/1), test_type
    """
    img_path = os.path.join(test_path, img_name)
    anno_path = os.path.join(test_path, img_name.replace(".jpg", ".json"))
    with open(anno_path) as f:
        anno = json.load(f)

    question = anno["question"]
    options = anno["options"]
    standard_answer = anno["options"][0]  # A 选项为标准答案
    standard_answer_with_prefix = "A. " + standard_answer

    # 选项字符串
    option_str = "\n" + "\n".join(f"{ABC_MAP[i + 1]}. {opt}" for i, opt in enumerate(options))

    # 处理图片
    pil_img = Image.open(img_path)
    base64_image = encode_pil_image_to_base64(pil_img)

    system_message = "You are a helpful assistant."
    user_text = INSTRUCTION_PROMPT_BEFORE.format(question=question, options=option_str)
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        {"type": "text", "text": user_text},
    ]

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
    }

    async with semaphore:
        resp = await eval_client.chat.completions.create(
            model=eval_model_name,
            messages=messages,
            **gen_kwargs,
        )
        response = resp.choices[0].message.content
        output_messages.append({"role": "assistant", "content": response})

    pred_output = extract_answer(response)
    # ========== 判题（先规则，必要时LLM） ==========
    acc_reward = rule_judge(pred_output, standard_answer_with_prefix)

    if acc_reward < 0:  # 交给 LLM 判定
        full_prompt = build_judge_prompt(pred_output, standard_answer_with_prefix, question)
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
            print("Judge Error: ", e)
            acc_reward = 0.0
            status = f"{status} | JudgeError: {e}"

    # 保存单条
    save_info = {
        "image": img_name,
        "question": question,
        "answer": standard_answer,
        "pred_output": pred_output,
        "response": response,
        "messages": output_messages,
        "status": status,
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
    vstar_bench_path: str,
    save_root: str,
    max_concurrency: int,
):
    test_path = os.path.join(vstar_bench_path, type_name)
    image_files = [f for f in os.listdir(test_path) if f.lower().endswith(".jpg")]
    total = len(image_files)

    # 输出目录
    model_name_safe = eval_model_name.replace("/", "_")
    judge_model_name_safe = judge_model_name.replace("/", "_")

    save_dir = os.path.join(
        save_root,
        f"{model_name_safe}_judge-{judge_model_name_safe}/{time_str}",
    )

    result_jsonl = os.path.join(save_dir, f"result_{type_name}.jsonl")
    raw_msg_jsonl = os.path.join(save_dir, f"result_{type_name}_raw_messages.jsonl")

    semaphore = asyncio.Semaphore(max_concurrency)

    # 进度条 + 实时准确率
    correct_count = 0
    processed = 0

    results_to_write = []
    raw_messages_to_write = []

    # tqdm 配置
    pbar = tqdm(total=total, desc=f"V* {type_name}", dynamic_ncols=True)

    async def _task(img_name: str):
        return await process_one_item(
            eval_client=eval_client,
            judge_client=judge_client,
            eval_model_name=eval_model_name,
            judge_model_name=judge_model_name,
            test_path=test_path,
            img_name=img_name,
            semaphore=semaphore,
        )

    for start in range(0, total, max_concurrency):
        batch = image_files[start : start + max_concurrency]
        tasks = [asyncio.create_task(_task(img)) for img in batch]
        for coro in asyncio.as_completed(tasks):
            res, acc_int = await coro
            results_to_write.append(res)
            raw_message = {
                "messages": res.pop("messages", []),
                "acc": acc_int,
                "answer": res.get("answer", ""),
            }
            raw_messages_to_write.append(raw_message)  # 与原一致：消息独立保存
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
            vstar_bench_path=args.vstar_bench_path,
            save_root=save_root,
            max_concurrency=args.num_workers,
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
            f"{eval_model_name.replace('/', '_')}_judge-{judge_model_name.replace('/', '_')}",
            time_str,
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
