import json
import os
import random
import re
from typing import Dict, Any, Optional

import requests
from openai import OpenAI

# OpenAI client configuration
openai_api_key = "EMPTY"
openai_api_base_list = [
    os.environ.get("LLM_AS_A_JUDGE_BASE", "http://28.12.131.187:8000/v1"),
]
print(f" [INFO] openai_api_base_list={openai_api_base_list}")

client_list = []
model_name_list = []

for api_base in openai_api_base_list:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
    )
    client_list.append(client)
    
    # Get model name
    try:
        response = requests.get(f"{api_base}/models")
        models = response.json()
        model_name_list.append(models["data"][0]["id"])
    except Exception as e:
        print(f" [WARNING] Failed to get model name from {api_base}: {e}")
        model_name_list.append("default-model")


def get_chat_template():
    """Get the chat template for accuracy evaluation."""
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template


def get_gpt4_score_ICE():
    """Get in-context examples for accuracy evaluation."""
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
"""

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
"""

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
"""

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
"""

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
"""

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
"""

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
"""

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


def get_accuracy_prompt(predict_str: str, ground_truth: str, question: str) -> str:
    """Generate prompt for accuracy evaluation."""
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


def get_consistency_prompt(think_process: str, final_answer: str) -> str:
    """Generate prompt for consistency evaluation between think process and answer."""
    consistency_template = '''
You are an expert about question answering. I will provide you a solution process and a final answer to the same question. Please evaluate the Consistency between the solution process and the final anwer: If the solution process draws the same conclusion with the final answer, rate the Consistency as 1, else rate 0.
Here is the solution process and the final answer for you to evaluate:\n

#### Solution Process: 
{gen_solution}\n

#### Final Answer: 
{gen_answer}\n

### Output Format (strictly follow)

Please provide an integer score to indicate the Consistency. Output the score in a JSON dictionary with nothing else for easy processing, in this form: 
```json
{{"Consistency": score}}.
```
Your Evaluation Result:
'''
    return consistency_template.format(gen_solution=think_process, gen_answer=final_answer)


def extract_answer_with_fallback(solution_str: str) -> tuple[str, bool]:
    """
    Extract answer from solution string using multiple fallback strategies.
    Returns (answer_text, is_format_error).
    """
    is_format_error = False
    answer_text = ""

    # Strategy 1: Try to extract from <answer> tags first
    predict_no_think = (
        solution_str.split("</think>")[-1].strip() if "</think>" in solution_str else solution_str.strip()
    )

    # Check <answer> tag format
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    # Try to extract from <answer> tags
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_no_think, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        # No proper <answer> tags found - this is a format error
        is_format_error = True

        # Strategy 2: If no <answer> tags, extract content after tool responses
        tool_response_match = re.search(
            r"</tool_response>\s*assistant\s*\n(.*?)$", predict_no_think, re.DOTALL | re.MULTILINE
        )
        if tool_response_match:
            answer_text = tool_response_match.group(1).strip()
        else:
            # Strategy 3: If no tool responses, look for content after </think>
            if "</think>" in solution_str:
                remaining_content = predict_no_think
                # Remove tool calls and responses
                remaining_content = re.sub(r"<tool_call>.*?</tool_call>", "", remaining_content, flags=re.DOTALL)
                remaining_content = re.sub(
                    r"<tool_response>.*?</tool_response>", "", remaining_content, flags=re.DOTALL
                )
                # Remove user/assistant markers
                remaining_content = re.sub(r"\b(user|assistant)\b", "", remaining_content)
                answer_text = remaining_content.strip()
            else:
                # Strategy 4: Use the entire solution_str as fallback
                answer_text = solution_str.strip()

    # Clean up answer text
    answer_text = answer_text.strip()

    # If answer is still empty after all strategies, mark as format error
    if not answer_text:
        is_format_error = True
        answer_text = solution_str.strip()  # Use full text as last resort

    return answer_text, is_format_error


def extract_think_process(solution_str: str) -> str:
    """Extract think process from <think></think> tags."""
    think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return ""


def check_format_strict(predict_str: str) -> tuple[bool, bool]:
    """
    Strict format checking based on reward_fn.py.
    Returns (is_format_error, give_tool_reward).
    """
    is_format_error = False
    give_tool_reward = False
    
    if predict_str.endswith("<|im_end|>"):
        predict_str = predict_str[: -len("<|im_end|>")]

    # Check think format pattern
    think_format_pattern = r"^<think>(?s:(?:(?!</think>).)*)</think>\n<answer>(?s:(?:(?!</answer>).)*)</answer>\Z"
    if not re.match(think_format_pattern, predict_str):
        is_format_error = True

    # Check vision tokens
    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    # Check think tags
    think_1 = predict_str.count("<think>")
    think_2 = predict_str.count("</think>")
    if think_1 != 1 or think_2 != 1:
        is_format_error = True

    # Check tool call tags
    tool_call_1 = predict_str.count("<tool_call>")
    tool_call_2 = predict_str.count("</tool_call>")
    if tool_call_1 != tool_call_2:
        is_format_error = True

    # Check tool response tags
    tool_response_1 = predict_str.count("<tool_response>")
    tool_response_2 = predict_str.count("</tool_response>")
    if tool_response_1 != tool_response_2:
        is_format_error = True

    if tool_response_1 != tool_call_1:
        is_format_error = True

    # Check answer tags
    count_answer_1 = predict_str.count("<answer>")
    count_answer_2 = predict_str.count("</answer>")
    if count_answer_1 != 1 or count_answer_2 != 1:
        is_format_error = True

    # Check if tool usage is valid for tool reward
    if (
        tool_call_1 > 0
        and tool_call_1 == tool_call_2
        and tool_response_1 == tool_response_2
        and tool_response_1 == tool_call_1
        and not is_format_error
    ):
        give_tool_reward = True

    return is_format_error, give_tool_reward


def make_llm_request_with_parsing(prompt: str, system_content: str = "You are a helpful assistant.", 
                                 temperature: float = 0.3, max_retries: int = 3, 
                                 parse_func=None, parse_args=None) -> Any:
    """
    Make LLM request with retry mechanism and parsing retry on parsing errors.
    
    Args:
        prompt: The user prompt to send
        system_content: The system message content
        temperature: Temperature for generation
        max_retries: Maximum number of retry attempts
        parse_func: Function to parse the response
        parse_args: Additional arguments for parse_func
        
    Returns:
        The parsed result, or None if all attempts failed
    """
    if not client_list or not model_name_list:
        print(" [WARNING] No client available for LLM request")
        return None
    
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]
    
    for attempt in range(max_retries):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                seed=random.randint(0, 1000000),
                temperature=temperature,
            )
            response = chat_response.choices[0].message.content.strip()
            
            # If no parsing function, return raw response
            if parse_func is None:
                return response
            
            # Try to parse the response
            try:
                if parse_args:
                    result = parse_func(response, *parse_args)
                else:
                    result = parse_func(response)
                return result
            except Exception as parse_error:
                print(f" [WARNING] Parsing attempt {attempt + 1}/{max_retries} failed: {parse_error}")
                if attempt == max_retries - 1:  # Last attempt
                    print(f" [ERROR] All {max_retries} attempts failed for parsing")
                    return None
                continue  # Try again for parsing error
                
        except Exception as e:
            print(f" [WARNING] LLM request attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:  # Last attempt
                print(f" [ERROR] All {max_retries} attempts failed for LLM request")
                return None
            continue  # Try again
    
    return None


def parse_accuracy_response(response: str) -> float:
    """Parse accuracy response from LLM."""
    # Parse response
    if "Judgement:" in response:
        response = response.split("Judgement:")[-1].strip()
        if "1" in response:
            return 1.0
        elif "0" in response:
            return 0.0
        else:
            print(f" [WARNING] Accuracy response format error response={response}")
            raise ValueError("Invalid response format")
    else:
        if response == "1":
            return 1.0
        elif response == "0":
            return 0.0
        elif response.startswith("1"):
            return 1.0
        elif response.startswith("0"):
            return 0.0
        else:
            print(f" [WARNING] Accuracy response format error response={response}")
            raise ValueError("Invalid response format")


def evaluate_accuracy(answer_text: str, ground_truth: str, question: str) -> float:
    """Evaluate accuracy using LLM judge."""
    full_prompt = get_accuracy_prompt(answer_text, ground_truth, question)
    
    result = make_llm_request_with_parsing(
        prompt=full_prompt, 
        temperature=0.1,
        parse_func=parse_accuracy_response
    )
    
    if result is None:
        return 0.0
    
    return result


def rule_math_verify(ground_truth: str, model_answer: str) -> bool:
    """Rule-based math verification using math_verify library."""
    try:
        # Import here to avoid circular import
        from math_verify import parse, verify

        # Set parsing_timeout=None to avoid signal.alarm() usage
        gold = parse(ground_truth, parsing_timeout=None)
        answer = parse(model_answer, parsing_timeout=None)
        return verify(gold, answer, timeout_seconds=None)
    except Exception as e:
        print(f" [ERROR math] rule_math_verify error: {e}")
        return False


def parse_math_response(response: str) -> bool:
    """Parse math verification response from LLM."""
    judgement = response.split("## Equivalence Judgement")[-1].lower()
    if "true" in judgement and "false" not in judgement:
        return True
    elif "false" in judgement and "true" not in judgement:
        return False
    else:
        print(" [ERROR math] verify bug output: ")
        raise ValueError("Invalid math response format")


def generative_math_verify(query: str, ground_truth: str, model_answer: str) -> bool:
    """Generative math verification using LLM judge."""
    full_prompt = MATH_VERIFY_PROMPT.format(
        query=query,
        gold_ans=ground_truth,
        pred_ans=model_answer,
    )

    result = make_llm_request_with_parsing(
        prompt=full_prompt, 
        system_content="", 
        temperature=0.1,
        parse_func=parse_math_response
    )
    
    if result is None:
        return False
    
    return result


def evaluate_math_accuracy(answer_text: str, ground_truth: str, question: str) -> float:
    """Evaluate math accuracy using rule-based verification first, then generative fallback."""
    # First try rule-based math verification
    if rule_math_verify(ground_truth, answer_text):
        return 1.0
    else:
        # Fallback to generative verification
        return 1.0 if generative_math_verify(question, ground_truth, answer_text) else 0.0


def parse_consistency_response(response: str) -> float:
    """Parse consistency response from LLM."""
    try:
        # Extract JSON from response
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            result = json.loads(json_str)
            consistency_score = result.get("Consistency", 0)
            return float(consistency_score)
        else:
            # Try to parse the entire response as JSON
            result = json.loads(response)
            consistency_score = result.get("Consistency", 0)
            return float(consistency_score)
    except (json.JSONDecodeError, ValueError) as e:
        print(f" [WARNING] Consistency response parsing error: {e}, response: {response}")
        raise  # Re-raise to trigger retry


def evaluate_consistency(think_process: str, final_answer: str) -> float:
    """Evaluate consistency between think process and final answer."""
    if not think_process or not final_answer:
        return 0.0

    full_prompt = get_consistency_prompt(think_process, final_answer)
    
    result = make_llm_request_with_parsing(
        prompt=full_prompt, 
        system_content="", 
        temperature=0.1,
        parse_func=parse_consistency_response
    )
    
    if result is None:
        return 0.0
    
    return result


def compute_enhanced_score(predict_str: str, ground_truth: str, extra_info: Optional[Dict[str, Any]] = None, is_math: bool = False) -> Dict[str, float]:
    """
    Enhanced reward computation combining multiple strategies.
    
    Args:
        predict_str: The solution string to be evaluated
        ground_truth: The ground truth answer for comparison
        extra_info: Additional information that might be needed for scoring
        is_math: Whether this is a math problem (uses math verification)
    
    Returns:
        Dict with keys: score, format_reward, tool_reward, acc_reward, consistency_reward
    """
    predict_str = predict_str.strip()
    
    # 1. Strict format checking (from reward_fn.py)
    is_format_error, give_tool_reward = check_format_strict(predict_str)
    
    # 2. Extract answer with fallback strategies (from deepeyes.py)
    answer_text, fallback_format_error = extract_answer_with_fallback(predict_str)
    
    # Combine format errors
    is_format_error = is_format_error or fallback_format_error
    
    # 3. Extract think process
    think_process = extract_think_process(predict_str)
    
    # 4. Evaluate accuracy based on problem type
    question_text = extra_info.get("question", "") if extra_info else ""
    if is_math:
        # Use math verification for math problems
        acc_reward = evaluate_math_accuracy(answer_text, ground_truth, question_text)
    else:
        # Use general accuracy evaluation for non-math problems
        acc_reward = evaluate_accuracy(answer_text, ground_truth, question_text)
    
    # 5. Evaluate consistency between think and answer
    consistency_reward = evaluate_consistency(think_process, answer_text)
    
    # 6. Penalize for excessively long answers (potential judge hacking)
    if answer_text and len(answer_text) >= 300:  # Use 300 as in original reward_fn.py
        is_format_error = True
    
    # 7. Calculate rewards
    format_reward = 0 if is_format_error else 1.0
    tool_reward = 1.0 if give_tool_reward else 0.0
    
    # 8. Final score calculation (using reward_fn.py weights + consistency)
    # Original: final_score = 1.0 * acc_reward + 0.25 * format_reward
    # Enhanced: Add consistency reward with weight 0.5
    final_score = 1.0 * acc_reward + 0.25 * format_reward + 0.5 * consistency_reward
    
    return {
        "score": final_score,
        "format_reward": format_reward,
        "tool_reward": tool_reward,
        "acc_reward": acc_reward,
        "consistency_reward": consistency_reward,
    }


def enhanced_reward_fn(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    sandbox_fusion_url: Optional[str] = None,
    concurrent_semaphore: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Enhanced reward function that combines the best of both reward_fn.py and deepeyes.py
    with additional consistency checking.
    
    Args:
        data_source: The source dataset identifier
        solution_str: The solution string to be evaluated
        ground_truth: The ground truth answer for comparison
        extra_info: Additional information that might be needed for scoring
        sandbox_fusion_url: URL for sandbox fusion (unused in this implementation)
        concurrent_semaphore: Semaphore for concurrent processing (unused in this implementation)
        
    Returns:
        Dict containing various reward components
    """
    # Determine if this is a math problem based on data source
    math_datasets = ["thinklite_eureka", "xince", "MathVista", "MathVerse"]
    is_math = data_source in math_datasets
    
    # Use enhanced scoring with math verification if needed
    result = compute_enhanced_score(solution_str, ground_truth, extra_info, is_math=is_math)
    
    return result


if __name__ == "__main__":
    # Test cases
    print("=== Enhanced Reward Function Test ===")
    
    # Test case 1: Well-formatted case with think and answer
    test_solution_1 = """<think>
I need to look at the image carefully to see where the woman is positioned relative to the man holding the camera.
Looking at the image, I can see a man holding a camera and a woman. The woman appears to be positioned on the left side of the man.
</think>
<answer>left</answer>"""
    
    test_ground_truth = "left"
    test_extra_info = {
        "question": "Is the woman to the left or to the right of the man who is holding the camera?"
    }
    
    print("Test case 1: Well-formatted with consistent think and answer")
    result1 = compute_enhanced_score(test_solution_1, test_ground_truth, test_extra_info)
    print(f"Result: {result1}")
    
    # Test case 2: Inconsistent think and answer
    test_solution_2 = """<think>
Looking at the image, the woman appears to be on the right side of the man holding the camera.
</think>
<answer>left</answer>"""
    
    print("\nTest case 2: Inconsistent think and answer")
    result2 = compute_enhanced_score(test_solution_2, test_ground_truth, test_extra_info)
    print(f"Result: {result2}")
    
    # Test case 3: Format error case
    test_solution_3 = """I think the woman is to the left of the man."""
    
    print("\nTest case 3: Format error (no think/answer tags)")
    result3 = compute_enhanced_score(test_solution_3, test_ground_truth, test_extra_info)
    print(f"Result: {result3}")
    
    # Test case 4: Math problem
    test_solution_4 = """<think>
I need to solve this quadratic equation: x^2 - 5x + 6 = 0
I can factor this as (x-2)(x-3) = 0
So x = 2 or x = 3
</think>
<answer>x = 2 or x = 3</answer>"""
    
    math_ground_truth = "x = 2, x = 3"
    math_extra_info = {
        "question": "Solve the quadratic equation x^2 - 5x + 6 = 0"
    }
    
    print("\nTest case 4: Math problem with consistent think and answer")
    result4 = compute_enhanced_score(test_solution_4, math_ground_truth, math_extra_info, is_math=True)
    print(f"Result: {result4}")
    
    # Test using the main reward function
    print("\n=== Testing enhanced_reward_fn ===")
    print("Math dataset test:")
    result_math = enhanced_reward_fn("MathVista", test_solution_4, math_ground_truth, math_extra_info)
    print(f"Math result: {result_math}")
    
    print("\nNon-math dataset test:")
    result_nonmath = enhanced_reward_fn("vstar", test_solution_1, test_ground_truth, test_extra_info)
    print(f"Non-math result: {result_nonmath}")
