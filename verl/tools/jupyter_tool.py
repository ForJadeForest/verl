# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import logging
import os
import socket
import threading
from contextlib import ExitStack
from enum import Enum
from io import BytesIO
from time import sleep
from typing import Any, Callable, Optional, TypeVar, Union
from uuid import uuid4

import ray
import ray.actor
import requests
from PIL import Image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


def get_local_ip():
    """
    获取本地IP地址
    :return: 本地IP地址字符串，或者在无法获取时返回 None
    """
    s = None
    try:
        # 创建一个UDP套接字
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 尝试连接到一个公共的IP地址（不需要是可达的）
        # 这里的8.8.8.8是Google的DNS服务器，端口号可以任意选择
        s.connect(('8.8.8.8', 80))
        # getsockname()返回套接字自己的地址
        ip = s.getsockname()[0]
    except Exception as e:
        logger.warning(f"Unable to get IP address: {e}")
        ip = None
    finally:
        if s:
            s.close()
    return ip


def get_default_sandbox_url():
    """Get default sandbox URL based on local IP."""
    default_sandbox_url = None
    local_ip = get_local_ip()
    if local_ip is None:
        return default_sandbox_url
    else:
        return f"http://{local_ip}:8080"


# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


def encode_image_path_base64(image_path):
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_pil_image_to_base64(pil_image):
    """Encode PIL Image to base64 string."""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def encode_image_base64(image: Union[str, Image.Image]) -> str:
    """Encode image to base64 string"""
    if isinstance(image, str):
        return encode_image_path_base64(image)
    elif isinstance(image, Image.Image):
        return encode_pil_image_to_base64(image)
    else:
        raise ValueError("Image must be a file path or PIL Image instance")


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image with automatic resizing if too small"""
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    
    # 验证图像尺寸
    width, height = image.size
    
    # 如果图像太小，进行resize操作
    if width < 28 or height < 28:
        logger.info(f"Image size {width}x{height} is smaller than 28, attempting resize.")
        
        # 计算resize比例，使最小边达到28像素
        min_dim = min(width, height)
        if min_dim == 0:  # 防止除零错误
            raise ValueError(f"Image has zero dimension: {width}x{height}")
        
        ratio = 28.0 / min_dim
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # 使用LANCZOS重采样进行resize（高质量）
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # 更新尺寸变量
        width, height = new_width, new_height
        image = resized_image
    
    # 检查宽高比不能>200
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > 200:
        raise ValueError(f"Image aspect ratio too extreme: {aspect_ratio:.2f}. Maximum allowed is 200.")
    
    return image


def run_jupyter_code(cell_list, sandbox_url, upload_file_dict=None, max_retries=2, retry_delay=1):
    """
    Run Jupyter code with retry mechanism.
    
    Args:
        cell_list: List of code cells to execute
        sandbox_url: URL of the sandbox service
        upload_file_dict: Optional dictionary of files to upload
        max_retries: Maximum number of retry attempts (default: 2)
        retry_delay: Delay between retries in seconds (default: 1)
    
    Returns:
        output_cells: List of output cells from Jupyter execution
    
    Raises:
        ValueError: If no output cells returned after all retries
        requests.RequestException: If all retry attempts fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            response = requests.post(
                f"{sandbox_url}/run_jupyter",
                json={
                    "cells": cell_list,
                    "kernel": "python3",
                    "files": upload_file_dict,
                    "total_timeout": 20,
                },
                timeout=22,  # Add request timeout
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            
            output_cells = response.json().get("cells", [])
            if not output_cells:
                raise ValueError(
                    f"No output cells returned from Jupyter execution. Cell List: {cell_list}\n"
                    f"Response: {response.json()}\n"
                    f"upload_file_dict: {upload_file_dict.keys() if upload_file_dict else None}"
                )
            
            return output_cells
            
        except (requests.RequestException, ValueError) as e:
            last_exception = e
            if attempt < max_retries:
                logger.debug(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                sleep(retry_delay)
                # Exponential backoff: increase delay for next retry
                retry_delay *= 2
            else:
                logger.debug(f"All {max_retries + 1} attempts failed. Last error: {e}")
    
    # If we get here, all retries failed
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError("All retry attempts failed with unknown error")


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
        error_message = f"[CODE RUN ERROR]: {e_name} - {e_value}\n\nPlease read the bug information and fix it to continue solve the question."

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
                logger.debug(f"Unknown key: {key}")
    text_output = ""
    if stdout:
        text_output += f"stdout: {stdout}\n"
    if display_text:
        text_output += f"display text: {display_text}\n"
    if error_message:
        text_output += f"error: {error_message}\n"

    image_output = display_image
    if "<image>" in text_output:
        logger.warning("Found <image> in text output, indicating image generation.")
        text_output = text_output.replace("<image>", "")
        logger.warning("Replaced <image> in text output with empty string.")
    return {
        "text_output": text_output.strip(),
        "image_output": image_output,
        "has_error": bool(error_message),
    }


def cell_output_to_str(cell_output: dict) -> dict:
    """Convert cell output to structured format."""
    text_output = cell_output["text_output"]
    image_output = cell_output["image_output"]
    if image_output:
        try:
            # 尝试转换所有图像，验证其尺寸
            valid_images = []
            error_infos = []
            for img in image_output:
                try:
                    valid_images.append(base64_to_image(img.split(",", 1)[-1]))
                except ValueError as e:
                    # 如果图像验证失败，记录错误但继续处理其他图像
                    logger.debug(f"Image validation failed: {e}")
                    error_infos.append(f"Image validation failed: {e}")
                    continue
            error_info_text = "\n".join(error_infos)
            if valid_images:
                return {
                    "text": text_output,
                    "images": valid_images,
                    "has_error": cell_output.get("has_error", False),
                }
            else:
                # 如果所有图像都无效，返回纯文本结果
                return {
                    "text": f"{text_output}\n[Processed Image Error: {error_info_text}]",
                    "images": [],
                    "has_error": cell_output.get("has_error", False)
                }
        except Exception as e:
            # 如果出现其他错误，返回纯文本结果
            logger.debug(f"Error processing images: {e}")
            return {
                "text": f"{text_output}\n[Error: Failed to process generated images. Error: {e}.\nPlease fix it first to show the image you want to check.]",
                "images": [],
                "has_error": cell_output.get("has_error", False)
            }
    else:
        return {
            "text": text_output,
            "images": [],
            "has_error": cell_output.get("has_error", False)
        }


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class JupyterExecutionWorker:
    """Worker for executing Jupyter operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10, sandbox_url=None):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None
        self.sandbox_url = get_default_sandbox_url()
        print(f" [INFO] {self.sandbox_url=}")

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="jupyter-rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing Jupyter code: {e}")
                    raise
        else:
            return fn(*fn_args, **fn_kwargs)


def init_jupyter_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode, sandbox_url=None
):
    """Initialize Jupyter execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(JupyterExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit, sandbox_url=sandbox_url)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class JupyterTool(BaseTool):
    """A tool for executing Python code in Jupyter environment.

    This tool provides code execution functionality with rate limiting and concurrent 
    execution support through Ray. It can handle images and provides structured output.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the Python code in Jupyter
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Example tool_schema:
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "excute_python_code_in_jupyter",
                "description": (
                    "Execute Python code in a persistent Jupyter environment to solve a wide variety of problems. "
                    "This powerful tool runs code and returns results and error information."
                    "\n\n**Persistent Environment**: This is a stateful Jupyter notebook environment where:\n"
                    "- Variables and data structures persist between code executions\n"
                    "- Previously imported libraries remain available for reuse\n"
                    "- Functions and classes you define are remembered\n"
                    "- You can build upon previous computations step by step\n"
                    "- Commonly used packages such as matplotlib, scipy, pandas, and seaborn are already installed\n"
                    "- You can get all output (including the image output) of jupyter cell. \n\n"
                    "Python code is incredibly versatile and can help you solve numerous types of problems:\n"
                    "1. **Mathematical & Scientific Computing**: Perform complex calculations, solve equations, statistical analysis, linear algebra operations using libraries like NumPy, SciPy, SymPy. If you are doing math question;\n"
                    "2. **Data Analysis & Visualization**: Process datasets, create charts and graphs, analyze trends using Pandas, Matplotlib, Plotly, Seaborn.\n"
                    "3. **Image Processing**: Load, manipulate, crop, rotate, enhance contrast, adjust brightness, apply filters, detect features using PIL. You can use img.show() to display results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute in the Jupyter environment.",
                        },
                    },
                    "required": ["code"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 20)
        self.rate_limit = config.get("rate_limit", 50)
        self.timeout = config.get("timeout", 30)
        self.sandbox_url = get_default_sandbox_url()
        print(f" [INFO] {self.sandbox_url=}")
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_jupyter_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
            sandbox_url=self.sandbox_url,
        )
        logger.info(f"Initialized JupyterTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Creates a new instance for Jupyter tool.

        This method initializes a new session for code execution, which can then be used
        for running Python code. It can optionally handle images that need to be uploaded
        to the Jupyter environment.

        Args:
            instance_id: An optional unique identifier for the instance. If not
                provided, a new UUID will be generated.
            **kwargs: Should contain 'image' key with image data, or 'create_kwargs'
                containing {'image': image_data}. Image can be one of the following:
                - A PIL.Image.Image object.
                - A string containing an HTTP or HTTPS URL.
                - A string containing a local file path.
                - A string containing a file URI (e.g., "file:///path/to/image.jpg").
                - A string containing a base64-encoded image in the format of "data:image/jpeg;base64,..."

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Initialize instance data
        instance_data = {
            "code_list": [],
            "upload_file_dict": {},
            "response": "",
            "reward": 0.0,
        }

        # Get image from kwargs if provided
        image = kwargs.get("image")
        if image is not None:
            try:
                from qwen_vl_utils import fetch_image
                img = fetch_image({"image": image})
                image_id = kwargs.get("image_id", f"image_{instance_id}")
                instance_data["upload_file_dict"][f"./{image_id}.jpg"] = encode_image_base64(img)
                instance_data["image"] = img
                logger.info(f"Image loaded for instance {instance_id}")
            except Exception as e:
                logger.info(f"Failed to load image for instance {instance_id}: {e}")
                raise ValueError(f"Failed to load image for instance {instance_id}: {e}")
        
        self._instance_dict[instance_id] = instance_data
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute Python code in Jupyter environment."""
        code = parameters.get("code")

        if not code or not isinstance(code, str):
            return (
                ToolResponse(text="Error: 'code' parameter is missing or not a string."),
                -0.05,
                {"success": False},
            )

        code = code.strip()
        if not code:
            return (
                ToolResponse(text="Error: 'code' parameter is empty."),
                -0.05,
                {"success": False},
            )

        instance_data = self._instance_dict[instance_id]
        instance_data["code_list"].append(code)

        try:
            # Execute the code using the execution pool
            cell_out = ray.get(
                self.execution_pool.execute.remote(
                    run_jupyter_code,
                    instance_data["code_list"],
                    self.sandbox_url,
                    instance_data["upload_file_dict"],
                )
            )

            # Parse the output
            parsed_output = parse_cell_output(cell_out[-1])
            code_output = cell_output_to_str(parsed_output)

            # Prepare response
            response_text = code_output["text"]
            response_images = code_output.get("images", [])
            has_error = code_output.get("has_error", False)

            if has_error:
                reward = -0.05
                success = False
            else:
                reward = 0.0
                success = True

            # Create tool response
            tool_response = ToolResponse(text=response_text)
            if response_images:
                tool_response.image = response_images

            return (
                tool_response,
                reward,
                {"success": success, "has_error": has_error},
            )

        except Exception as e:
            logger.error(f"Error executing Jupyter code: {e}")
            return (
                ToolResponse(text=f"Error executing Python code: {e}"),
                -0.05,
                {"success": False, "error": str(e)},
            )

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
