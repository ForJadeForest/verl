import base64
import json
import re
import time
from io import BytesIO

import requests
from openai import AsyncOpenAI
from PIL import Image

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 4096 * 2 * 28 * 28


def get_upload_image_prompt(upload_img_paths: list | str) -> str:
    """Get the prompt for uploading an image."""
    if isinstance(upload_img_paths, str):
        upload_img_paths = [upload_img_paths]
    img_path_text = ""
    for i, img_path in enumerate(upload_img_paths):
        img_path_text += f'Picture {i} path: "{img_path}"\n'

    question_prefix = f"I have upload the following images:\n{img_path_text}\n\n"
    return question_prefix


def query_template(question: str, sandbox_image_path: list[str] | str, need_picture_text: bool = True):
    if isinstance(sandbox_image_path, str):
        sandbox_image_path = [sandbox_image_path]
    image_text = ''.join([f'Picture {i}: <image>\n' for i in range(len(sandbox_image_path))]) if need_picture_text else ""
    return f"""{get_upload_image_prompt(sandbox_image_path)}{image_text}
Now please answer the following question:
{question}
Please answer in the following format:
<think>...</think>
<answer>...</answer>""".strip()



def run_jupyter_code(cell_list, sandbox_url, upload_file_dict=None):
    attempts = 3
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                f"{sandbox_url}/run_jupyter",
                json={
                    "cells": cell_list,
                    "kernel": "python3",
                    "files": upload_file_dict,
                    "total_timeout": 22,
                },
            )
            try:
                data = response.json()
            except Exception as e:
                data = {}
                last_err = f"invalid_json_attempt_{attempt}: {e}"

            output_cells = data.get("cells", []) if isinstance(data, dict) else []
            if output_cells:
                return output_cells
            last_err = f"no_output_cells_attempt_{attempt}"
        except Exception as e:
            last_err = f"request_error_attempt_{attempt}: {e}"
        if attempt < attempts:
            time.sleep(0.5)

    return None




def parse_cell_output(cell_output: dict) -> dict:
    """Parse the output of a Jupyter cell."""
    if not cell_output:
        return {
            "text_output": "",
            "image_output": [],
            "has_error": False,
        }
    stdout = cell_output.get("stdout", "")

    errors = cell_output.get("error", "")
    error_message = ""
    for e in errors:
        # traceback = e.get("traceback", "")
        e_name = e.get("ename", "")
        e_value = e.get("evalue", "")
        error_message = f"[CODE RUN ERROR]: {e_name} - {e_value}\n\n"
        error_message += "Please read the bug information and fix it to continue solve the question."
        error_message += "Hint: All variables in this cell can not be used in the next cell."

        if len(error_message) > 3000:
            error_message = error_message[:1500] + "..." + error_message[-1500:]
    # show display output
    display_output = cell_output.get("display", [])
    display_text = ""
    display_image = []
    for cell_output_item in display_output:
        for key, value in cell_output_item.items():
            if key == "text/plain":
                display_text += value
            elif key == "image/png":
                display_image.append(f"data:image/png;base64,{value}")
            elif key == "image/jpeg":
                display_image.append(f"data:image/jpeg;base64,{value}")
            else:
                print(f"Unknown key: {key}")
    text_output = ""
    if stdout:
        text_output += f"stdout: {stdout}\n"
    if display_text:
        text_output += f"display text: {display_text}\n"
    if error_message:
        text_output += f"error: {error_message}\n"

    image_output = display_image
    return {
        "text_output": text_output.strip(),
        "image_output": image_output,
        "has_error": bool(error_message),
    }

def process_image_output(image_output: list[str]) -> list:
    from evaluate.utils import encode_image_base64
    
    # 修复变量名冲突
    decoded_images = [base64.b64decode(image.split(",", 1)[-1]) for image in image_output]
    pil_images = [Image.open(BytesIO(img_data)).convert("RGB") for img_data in decoded_images]
    
    def check_image_size(image: Image.Image) -> str:  # 修正返回类型
        width, height = image.size
        if width < 28 or height < 28:
            print(f"Image size {width}x{height} is smaller than 28, attempting resize.")
            
            # 计算resize比例，使最小边达到28像素
            min_dim = min(width, height)
            if min_dim == 0:  # 防止除零错误
                raise ValueError(f"Image has zero dimension: {width}x{height}")
            
            ratio = 28.0 / min_dim
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # 使用LANCZOS重采样进行resize（高质量）
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # 编码图像并返回
        encoded_image = f"data:image/jpeg;base64,{encode_image_base64(image)}"
        return encoded_image
    
    processed_images = [check_image_size(image) for image in pil_images]
    return processed_images

def cell_output_to_content_list(cell_output: dict) -> list:
    text_output = cell_output["text_output"]
    image_output = cell_output["image_output"]
    content = [
        {"type": "text", "text": "<tool_response>\n<interpreter>" + text_output},
    ]
    image_output = process_image_output(image_output)
    for image in image_output:
        content.append(
            {"type": "image_url", "image_url": {"url": image}}  # type: ignore
        )
    content.append({"type": "text", "text": "</interpreter>\n</tool_response>"})
    return content

def extract_code_from_response(response_text: str) -> str:
    """Extract code from the response."""
    if "<tool_call>" not in response_text or "</tool_call>" not in response_text:
        raise ValueError("Response does not contain a valid tool call.")

    code_json_str = response_text.split("<tool_call>")[1].split("</tool_call>")[0]
    try:
        code = json.loads(code_json_str)["arguments"]["code"]
    except (json.JSONDecodeError, KeyError):
        raise ValueError("Failed to extract code from the tool call.")
    return code


def call_tool(
    resposne: str,
    code_list: list,
    upload_file_dict: dict,
    sandbox_url: str | None = None,
) -> dict:
    """Call the tool with the extracted code."""

    try:
        code = extract_code_from_response(resposne)
    except Exception as e:
        return {"status": "error", "message": "Extract Code Error: " + str(e)}
    code_list.append(code)

    try:
        code_output = run_jupyter_code(code_list, sandbox_url, upload_file_dict=upload_file_dict)
    except Exception as e:
        print("Call Tool Error: ", e)
        return {"status": "error", "message": "Tool Call Jupyter Runing Error: " + str(e)}
    if code_output is None:
        print("Call Tool Error: No output cells returned from Jupyter execution")
        return {"status": "error", "message": "Tool Call Jupyter Runing Error: No output cells returned from Jupyter execution"}

    parsed_output = parse_cell_output(code_output[-1])
    code_output_content_list = cell_output_to_content_list(parsed_output)

    return {
        "status": "success",
        "code_output_content_list": code_output_content_list,
        "code": code,
        "code_output": code_output,
        "code_list": code_list,
    }


async def solve_one_query(
    message,
    upload_image_dict: dict,
    client: AsyncOpenAI,
    model_name: str,
    sandbox_url: str,
    max_turn=10,
    gen_kwargs=None,
) -> list[dict]:
    """Solve one query with the given message."""
    if gen_kwargs is None:
        gen_kwargs = {}

    code_list = []
    response_num = 0

    while response_num < max_turn:
        params = {
            "model": model_name,
            "messages": message,
            **gen_kwargs,
        }
        response = await client.chat.completions.create(**params)
        response_text = response.choices[0].message.content

        if "<tool_call>" in response_text and "</tool_call>" in response_text:
            tool_out = call_tool(response_text, code_list, upload_image_dict, sandbox_url)
            if tool_out["status"] == "error":
                print("Tool Call Error: ", tool_out["message"])
                message.extend(
                    [
                        {"role": "assistant", "content": response_text},
                        {
                            "role": "user",
                            "content": f"Error: {tool_out['message']}",
                        },
                    ]
                )
            else:
                code_list = tool_out["code_list"]
                code_output_content_list = tool_out["code_output_content_list"]
                message.extend(
                    [
                        {"role": "assistant", "content": response_text},
                        {"role": "user", "content": code_output_content_list},
                    ]
                )

            if response_num == max_turn - 2:
                if isinstance(message[-1]["content"], list):
                    message[-1]["content"].append(
                        {
                            "type": "text",
                            "text": "You have reached the maximum number of tool calls. Please provide a final response.",
                        }
                    )
                elif isinstance(message[-1]["content"], str):
                    message[-1]["content"] += (
                        "\nYou have reached the maximum number of tool calls. Please provide a final response."
                    )
                else:
                    raise ValueError(f"Unknown message content type: {type(message[-1]['content'])}")
        else:
            message.append({"role": "assistant", "content": response_text})
            break
        response_num += 1

    return message


def check_is_done(response_text: str) -> bool:
    """Helper function to check if the LLM response is a final answer."""
    return "<tool_call>" not in response_text and "</tool_call>" not in response_text


def extract_response(messages: list[dict]) -> str:
    response = ""
    
    assistant_begin = False
    for m in messages:
        # find the first assistant message
        if not assistant_begin and m['role'] == "assistant":
            assistant_begin = True
        if not assistant_begin:
            continue

        content = m.get("content", "")
        if isinstance(content, str):
            response += content
        elif isinstance(content, list):
            for c in content:
                if c.get("type") == "text":
                    response += c.get("text", "")
                elif c.get("type") == "image_url":
                    response += "<IMAGE>"
    return response

def extract_answer(response: str) -> str:
    search_result = re.search(r"<answer>(.*)</answer>", response, re.DOTALL)
    if search_result:
        response = search_result.group(1)
    else:
        response = response.strip()
    return response

def build_user_message_for_gen(
    base64_images: str | list[str], 
    question_text: str
) -> list[dict]:
    # 统一成列表并复制，避免修改调用方的原始列表
    images = [base64_images] if isinstance(base64_images, str) else list(base64_images)

    parts = question_text.split("<image>")
    assert len(images) == len(parts) - 1, (
        "The number of images and the number of <image> tags in the question text must be the same"
        f"images: {len(images)}, parts: {len(parts)}"
    )

    content: list[dict] = []
    for i, part in enumerate(parts):
        if part:  # 保留空段行为：如果连续 <image>，就只插图，不插文本
            content.append({"type": "text", "text": part})
        if i < len(images):  # 只在段与段之间插入对应图片（不会误给最后一段再塞一张）
            img = images[i]
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"},
            })
    return content
