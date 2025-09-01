
import json
import os
import random
import re

import requests
from openai import OpenAI

Box = list[float]  # [x1, y1, x2, y2]


openai_api_key = "EMPTY"
openai_api_base_list = [
    os.environ.get("LLM_AS_A_JUDGE_BASE", "http://10.39.13.134:18901/v1"),
]

if not openai_api_base_list[0]:
    raise ValueError("LLM_AS_A_JUDGE_BASE is not set")

print(f" [INFO] {openai_api_base_list=}")
client_list = []
for api_base in openai_api_base_list:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
    )
    client_list.append(client)
model_name_list = []
for client in client_list:
    response = requests.get(f"{api_base}/models")
    models = response.json()
    model_name_list.append(models["data"][0]["id"])



def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
"""  # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
"""  # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
"""  # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
"""  # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
"""  # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
"""  # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
"""  # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


COMMON_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level reasoning problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Semantic Equivalence: Carefully examine the expression in both answers. Confirm whether the semantic meaning of student's final answer is equivalent to the reference answer, even when expressed with different wording or format.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


def get_prompt(predict_str: str, ground_truth: str, question: str) -> str:
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f"{demo_prompt}{test_prompt}"

    return full_prompt


def extract_answer(text) -> str:
    """
    从给定的文本中提取<answer></answer>标签内部的内容。

    参数:
        text (str): 包含<answer>标签的文本

    返回:
        str: 最后一个标签内部的内容，如果未找到则返回空字符串。
    """
    # 使用非贪婪模式匹配所有<answer>和</answer>之间的内容
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()  # 返回最后一个匹配
    return ""


def check_format(predict_str):
    is_format_error = False
    give_tool_reward = False
    if predict_str.endswith("<|im_end|>"):
        predict_str = predict_str[: -len("<|im_end|>")]

    # think_format_pattern = r"^<think>(?s:(?:(?!</think>).)*)</think>\n{1,2}<answer>(?s:(?:(?!</answer>).)*)</answer>\Z"
    # if not re.match(think_format_pattern, predict_str):
    #     is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    # think_1 = predict_str.count("<think>")
    # think_2 = predict_str.count("</think>")

    # if think_1 != 1 or think_2 != 1:
    #     is_format_error = True

    tool_call_1 = predict_str.count("<tool_call>")
    tool_call_2 = predict_str.count("</tool_call>")
    if tool_call_1 != tool_call_2:
        is_format_error = True

    tool_response_1 = predict_str.count("<tool_response>")
    tool_response_2 = predict_str.count("</tool_response>")
    if tool_response_1 != tool_response_2:
        is_format_error = True

    if tool_response_1 != tool_call_1:
        is_format_error = True

    count_answer_1 = predict_str.count("<answer>")
    count_answer_2 = predict_str.count("</answer>")
    if count_answer_1 != 1 or count_answer_2 != 1:
        is_format_error = True

    if (
        tool_call_1 > 0
        and tool_call_1 == tool_call_2
        and tool_response_1 == tool_response_2
        and tool_response_1 == tool_call_1
        and not is_format_error
    ):
        give_tool_reward = True

    return is_format_error, give_tool_reward


def compute_code_panelty(predict_str: str) -> float:
    code_error_count = predict_str.count("[CODE RUN ERROR]")
    tool_call_num = predict_str.count("<tool_response>")

    if tool_call_num > 0:
        return - code_error_count / tool_call_num
    else:
        return 0.0


def compute_repetition_penalty(predict_str: str) -> float:
    if "</tool_call><tool_call>" in predict_str:
        return max(-1, -0.25 * predict_str.count("</tool_call><tool_call>"))
    else:
        return 0.0
    



def iou(box_a: Box, box_b: Box) -> float:
    """计算两个矩形框的 IoU。假设输入合法: x1<=x2, y1<=y2"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def is_valid_box(b: Box) -> bool:
    """检查 box 合法性 (x1<=x2, y1<=y2)。"""
    x1, y1, x2, y2 = b
    return (x1 <= x2) and (y1 <= y2)

def calculate_grounding_reward(pred_bbox: list[Box], gt_bbox: list[Box]) -> tuple[float, float, float, float]:
    """
    其中 R_IoU = 0.5*(R_IoU^R + R_IoU^P)
         R_IoU^R = (1/M) * sum_k max_i IoU( b̂_i, b_k )
         R_IoU^P = (1/N) * sum_i max_k IoU( b̂_i, b_k )

    规则：
      - 若 pred_bbox 中任意 box 非法(x1>x2 或 y1>y2)，则整个 reward=0
    """
    # 检查合法性
    if any(not is_valid_box(b) for b in pred_bbox):
        return 0.0, 0.0, 0.0

    N = len(pred_bbox)
    M = len(gt_bbox)

    # Recall term
    if M == 0:
        R_recall = 0.0
    else:
        R_recall = sum(max(iou(p, g) for p in pred_bbox) if pred_bbox else 0.0 for g in gt_bbox) / M

    # Precision term
    if N == 0:
        R_precision = 0.0
    else:
        R_precision = sum(max(iou(p, g) for g in gt_bbox) if gt_bbox else 0.0 for p in pred_bbox) / N

    R_iou = 0.5 * (R_recall + R_precision)
    return R_iou, R_recall, R_precision



def compute_score(predict_str: str, ground_truth: str, extra_info=None) -> dict:
    predict_str = predict_str.strip()
    is_format_error, give_tool_reward = check_format(predict_str)

    answer_text = extract_answer(predict_str)
    if not answer_text:
        is_format_error = True
        answer_text = predict_str
    else:
        answer_text = answer_text.strip()

    question_text = extra_info["question"]
    full_prompt = get_prompt(answer_text, ground_truth, question_text)

    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt},
        ],
        seed=random.randint(0, 1000000),
        temperature=0.3,
    )
    response = chat_response.choices[0].message.content.strip()
    if "Judgement:" in response:
        response = response.split("Judgement:")[-1].strip()
        if "1" in response:
            acc_reward = 1.0
        elif "0" in response:
            acc_reward = 0.0
        else:
            print(f" [WARNING] resp format error {response=}")
            acc_reward = 0.0
    else:
        if response == "1":
            acc_reward = 1.0
        elif response == "0":
            acc_reward = 0.0
        elif response.startswith("1"):
            acc_reward = 1.0
        elif response.startswith("0"):
            acc_reward = 0.0
        else:
            print(f" [WARNING] resp format error {response=}")
            acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if answer_text and len(answer_text) >= 300:
        is_format_error = True

    format_reward = 0 if is_format_error else 1.0
    code_panelty = compute_code_panelty(predict_str)
    code_repetition_reward = compute_repetition_penalty(predict_str)
    final_score = 1.0 * acc_reward + 0.25 * format_reward + code_panelty + code_repetition_reward

    return {
        "score": final_score,
        "format_reward": format_reward,
        "acc_reward": acc_reward,
        "acc": acc_reward,
        "code_error_reward": code_panelty,
        "code_repetition_reward": code_repetition_reward,
    }

def compute_ground_score(predict_str: str, ground_truth: str, extra_info=None) -> dict:
    predict_str = predict_str.strip()
    is_format_error, give_tool_reward = check_format(predict_str)

    answer_text = extract_answer(predict_str)
    if not answer_text:
        is_format_error = True
        answer_text = predict_str
    else:
        answer_text = answer_text.strip()

    question_text = extra_info["question"]
    full_prompt = get_prompt(answer_text, ground_truth, question_text)

    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt},
        ],
        seed=random.randint(0, 1000000),
        temperature=0.3,
    )
    response = chat_response.choices[0].message.content.strip()
    if "Judgement:" in response:
        response = response.split("Judgement:")[-1].strip()
        if "1" in response:
            acc_reward = 1.0
        elif "0" in response:
            acc_reward = 0.0
        else:
            print(f" [WARNING] resp format error {response=}")
            acc_reward = 0.0
    else:
        if response == "1":
            acc_reward = 1.0
        elif response == "0":
            acc_reward = 0.0
        elif response.startswith("1"):
            acc_reward = 1.0
        elif response.startswith("0"):
            acc_reward = 0.0
        else:
            print(f" [WARNING] resp format error {response=}")
            acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if answer_text and len(answer_text) >= 300:
        is_format_error = True


    def extract_bbox(predict_str) -> list[tuple[float, float, float, float]]:
        pattern = r"<box>(.*?)</box>"
        matches = re.findall(pattern, predict_str, re.DOTALL)
        bboxs = []
        for match in matches:
            bbox = match.strip()
            bbox = json.loads(bbox)
            bboxs.append(bbox)
        return bboxs

    try:
        pred_bboxs = extract_bbox(predict_str)
    except Exception as e:
        print(f" [Extract Bbox ERROR] extract_bbox error: {e}")
        pred_bboxs = []

    gt_bboxs = extra_info["bbox"]
    if isinstance(gt_bboxs, str):
        gt_bboxs = json.loads(gt_bboxs)
        gt_bboxs = [item["bbox"] for item in gt_bboxs]
    if isinstance(gt_bboxs, list) and not isinstance(gt_bboxs[0], list):
        gt_bboxs = [gt_bboxs]

    try:
        ground_reward, R_recall, R_precision = calculate_grounding_reward(pred_bboxs, gt_bboxs)
    except Exception as e:
        print(f" [Calculate Grounding Reward ERROR] calculate_grounding_reward error: {e}")
        ground_reward = 0.0
        R_recall = 0.0
        R_precision = 0.0

    format_reward = 0 if is_format_error else 1.0
    code_panelty = compute_code_panelty(predict_str)
    code_repetition_reward = compute_repetition_penalty(predict_str)
    if "<answer>" not in predict_str:
        ground_reward = 0.0
    
    final_score = acc_reward + 0.5 * format_reward + ground_reward + code_panelty + code_repetition_reward
    
    return {
        "score": final_score,
        "format_reward": format_reward,
        "acc_reward": acc_reward,
        "ground_reward": ground_reward,
        "ground_recall_reward": R_recall,
        "ground_precision_reward": R_precision,
        "acc": acc_reward,
        "code_error_reward": code_panelty,
        "code_repetition_reward": code_repetition_reward,
    }


def reward_fn(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """

    if data_source in ["vstar", "vl_agent", "chart"] or data_source.startswith(
        "detailedbench"
    ):
        res = compute_score(solution_str, ground_truth, extra_info)

    elif data_source in ["Ground-R1", "TreeVGR-RL"]:
        res = compute_ground_score(solution_str, ground_truth, extra_info)
    elif data_source in ["MathVista", "MathVerse", "HallusionBench", "MMMU", "VStar", "AI2D", "HRBench-4k", "HRBench-8k", "TreeBench", "MMStar"]:
        res = compute_score(solution_str, ground_truth, extra_info)
    else:
        raise NotImplementedError(
            f"Reward function is not implemented for {data_source=}"
        )

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
