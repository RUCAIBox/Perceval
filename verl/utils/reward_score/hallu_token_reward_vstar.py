import requests
import random
import re
import os
import json
from typing import List, Dict, Any, Tuple, Optional, Union
from openai import OpenAI
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from dataclasses import dataclass
import torch
import base64
import ast

from math_verify import parse, verify

try:
    from openai import APIError, RateLimitError, APITimeoutError
    # 将可能发生的、可重试的异常类型放在一个元组里
    RETRYABLE_EXCEPTIONS = (APIError, RateLimitError, APITimeoutError)
except ImportError:
    # 如果没有安装 openai 库或库版本不同，就捕获通用异常
    RETRYABLE_EXCEPTIONS = (Exception,)

from perceval.prompts import (
    gpt_prompt,
    claude_prompt,
    claude_input_prompt,
    claude_tool_prompt,
    claude_hallu_and_consistency_prompt,
    hallu_and_consistency_with_observation,
)

"""
Input: data_source: str, question: str, response: str, extra_info: Optional[Dict] = None
Output: rule-based score, process_correct_indicator, [list of sub-sentence-region]
总函数: compute_score
核心功能: 
1. compute_acc 计算rule-based score
2. result_verify 发送给outcome reward model来判断answer的正确性（比对answer和ground_truth） 
3. process_verify 发送给process reward model来判断response中有问题的部分
4. locate_problems 根据process_verify的结果定位 sub-sentence-region    """

MAX_RETRIES=4

COMPILED_PATTERN_WHITESPACE = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL)

def check_format_by_regex_whitespace(s: str) -> bool:
    """
    使用正则表达式检查格式，允许标签间有空白字符。
    这是处理这种灵活间隔的最佳方法。

    Args:
        s: 待检查的字符串。

    Returns:
        如果字符串符合格式则返回 True，否则返回 False。
    """
    return COMPILED_PATTERN_WHITESPACE.fullmatch(s) is not None


def initialize_clients(
    api_base_list: List[str],
    client_description: str
) -> Tuple[List[OpenAI], List[str]]:
    """
    Initializes OpenAI clients and fetches model names from a list of base URLs.

    Args:
        api_base_list: A list of base URLs for the API endpoints.
        client_description: A description for logging purposes (e.g., "Judge").

    Returns:
        A tuple containing:
        - A list of initialized OpenAI client instances.
        - A list of model names discovered from the endpoints.
    """
    # Define the API key
    api_key = "EMPTY"

    print(f"{client_description} API Base List: {api_base_list}")

    # Initialize lists to hold the client instances and model names.
    client_list = []
    model_name_list = []

    # Create a client for each API base URL
    for api_base in api_base_list:
        if not api_base:  # Skip if the URL is empty or None
            continue
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        client_list.append(client)

    # Fetch model names from each client's endpoint
    for client in client_list:
        try:
            # Use the base_url from the client object to make the request
            api_base_url = str(client.base_url)
            response = requests.get(f"{api_base_url}models")
            response.raise_for_status()  # Checks for HTTP errors
            models = response.json()

            # Safely access the model name
            if models.get('data') and isinstance(models['data'], list) and len(models['data']) > 0:
                model_name_list.append(models['data'][0]['id'])
            else:
                print(f"No model data found in response from {api_base_url}")

        except requests.RequestException as e:
            # Handle network-related errors (e.g., connection refused)
            print(f"Error fetching models from {client.base_url}: {e}")
        except (KeyError, IndexError, TypeError):
            # Handle cases where the JSON response is not in the expected format
            print(f"Invalid or unexpected response format from {client.base_url}")

    print(f"Detected {client_description} Models: {model_name_list}")
    return client_list, model_name_list

# --- Main Execution ---

# Define the list of base URLs for the Judge endpoints.
# You can add more URLs to this list as needed.
judge_api_base_str = os.environ.get('LLM_AS_A_JUDGE_BASE', "http://0.0.0.0:9999/v1 http://0.0.0.0:12298/v1")
prm_api_base_str = os.environ.get('PRM_BASE', "http://0.0.0.0:3456/v1 http://0.0.0.0:2789/v1")

def get_list_from_str(url_str):
    url_list = []
    if url_str:
    # 使用 .split(' ') 将字符串按空格分割成一个列表
        url_list = url_str.split(' ')
    else:
        print("环境变量 'URL_LIST' 未设置。")
    return url_list

judge_api_base_list = get_list_from_str(judge_api_base_str)
process_reward_api_base_list = get_list_from_str(prm_api_base_str)

# Initialize Judge clients and models
judge_client_list, judge_model_name_list = initialize_clients(
    api_base_list=judge_api_base_list,
    client_description="Judge"
)

print("\n" + "="*50 + "\n")

# Define the list of base URLs for the Process Reward endpoints.
# process_reward_api_base_list = [
#     os.environ.get("PROCESS_REWARD_BASE", "http://0.0.0.0:18902/v1"),
# ]

# Initialize Process Reward clients and models
process_reward_client_list, process_reward_model_name_list = initialize_clients(
    api_base_list=process_reward_api_base_list,
    client_description="Process Reward"
)



@dataclass
class TokenPosition:
    """Token位置结果"""
    token_start: int
    token_end: int
    status: str  # "success", "not_found", "error"
    error_msg: Optional[str] = None


class SerialTokenPositionFinder:
    """最简单高效的串行Token位置查找器"""
    
    def __init__(self, tokenizer_name_or_path: str):
        """
        初始化查找器
        
        Args:
            tokenizer_name_or_path: tokenizer名称或路径
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def find_substring_positions(self, full_text: str, target_substrings: List[str]) -> List[TokenPosition]:
        """
        在单个文本中查找多个子串的token位置
        
        Args:
            full_text: 完整文本
            target_substrings: 目标子串列表
            
        Returns:
            List[TokenPosition]: 每个子串对应的token位置
        """
        if not target_substrings:
            return []
        
        results = []
        
        try:
            # 关键优化：只tokenize一次
            encoding = self.tokenizer(
                full_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=False,
                return_tensors=None  # 避免tensor开销
            )
            
            offsets = encoding['offset_mapping']
            
            # 为每个子串查找位置
            for substring in target_substrings:
                result = self._find_single_substring(full_text, substring, offsets)
                results.append(result)
                
        except Exception as e:
            # 如果tokenize失败，为所有子串返回错误
            for _ in target_substrings:
                results.append(TokenPosition(
                    token_start=-1,
                    token_end=-1,
                    status="error",
                    error_msg=f"Tokenization failed: {str(e)}"
                ))
        
        return results

    def _find_single_substring(self, full_text: str, substring: str, 
                             offsets: List[Tuple[int, int]]) -> TokenPosition:
        """
        在已tokenized的文本中查找单个子串
        
        Args:
            full_text: 完整文本
            substring: 目标子串
            offsets: tokenizer的offset_mapping
            
        Returns:
            TokenPosition: 查找结果
        """
        try:
            # 1. 查找字符位置
            char_start = full_text.find(substring)
            if char_start == -1:
                return TokenPosition(
                    token_start=-1,
                    token_end=-1,
                    status="not_found"
                )
            
            char_end = char_start + len(substring)
            
            # 2. 转换为token位置
            token_start = None
            token_end = None
            
            for i, (start, end) in enumerate(offsets):
                # 找到包含char_start的token
                if token_start is None and start <= char_start < end:
                    token_start = i
                
                # 找到包含或超过char_end的token
                if start < char_end <= end:
                    token_end = i + 1
                    break
                elif start == char_end:
                    token_end = i
                    break
            
            # 3. 边界处理
            if token_start is None:
                # 如果没找到包含char_start的token，找最接近的
                for i, (start, end) in enumerate(offsets):
                    if start >= char_start:
                        token_start = i
                        break
                else:
                    token_start = len(offsets)
            
            if token_end is None:
                # 如果没找到包含char_end的token，找最接近的
                for i, (start, end) in enumerate(offsets):
                    if start >= char_end:
                        token_end = i
                        break
                else:
                    token_end = len(offsets)
            
            # 4. 验证结果
            if token_start is not None and token_end is not None and token_start <= token_end:
                return TokenPosition(
                    token_start=token_start,
                    token_end=token_end,
                    status="success"
                )
            else:
                return TokenPosition(
                    token_start=-1,
                    token_end=-1,
                    status="error",
                    error_msg="Invalid token mapping"
                )
                
        except Exception as e:
            return TokenPosition(
                token_start=-1,
                token_end=-1,
                status="error",
                error_msg=str(e)
            )

    def find_batch_positions(self, full_texts: List[str], 
                           target_substrings_list: List[List[str]]) -> List[List[TokenPosition]]:
        """
        批量处理多个文本
        
        Args:
            full_texts: 完整文本列表
            target_substrings_list: 每个文本对应的目标子串列表
            
        Returns:
            List[List[TokenPosition]]: 每个文本的查找结果列表
        """
        if len(full_texts) != len(target_substrings_list):
            raise ValueError("full_texts和target_substrings_list长度必须相同")
        
        batch_results = []
        
        for full_text, target_substrings in zip(full_texts, target_substrings_list):
            text_results = self.find_substring_positions(full_text, target_substrings)
            batch_results.append(text_results)
        
        return batch_results

    def verify_results(self, full_text: str, substring: str, 
                      token_start: int, token_end: int) -> bool:
        """
        验证查找结果的正确性
        
        Args:
            full_text: 原始文本
            substring: 目标子串
            token_start: 找到的token起始位置
            token_end: 找到的token结束位置
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 获取找到的tokens并解码
            encoding = self.tokenizer(full_text, add_special_tokens=False, return_tensors=None)
            tokens = encoding['input_ids']
            
            if token_start < 0 or token_end < 0 or token_start >= len(tokens) or token_end > len(tokens):
                return False
            
            found_tokens = tokens[token_start:token_end]
            decoded_text = self.tokenizer.decode(found_tokens, skip_special_tokens=True)
            
            # 检查是否包含目标子串
            return substring in decoded_text or decoded_text in substring
            
        except Exception:
            return False


def format_results(results: List[List[TokenPosition]], 
                  full_texts: List[str], 
                  target_substrings_list: List[List[str]]) -> None:
    """格式化打印结果"""
    print("\n=== 查找结果 ===")
    
    for text_idx, (text_results, full_text, target_substrings) in enumerate(
        zip(results, full_texts, target_substrings_list)
    ):
        print(f"\n文本 {text_idx + 1} (长度: {len(full_text)} 字符):")
        print(f"  文本预览: '{full_text[:100]}{'...' if len(full_text) > 100 else ''}'")
        
        for substring_idx, (result, substring) in enumerate(zip(text_results, target_substrings)):
            status_symbol = "✓" if result.status == "success" else "✗"
            print(f"  {status_symbol} 子串 {substring_idx + 1}: '{substring}'")
            
            if result.status == "success":
                print(f"    Token位置: [{result.token_start}, {result.token_end})")
            else:
                print(f"    状态: {result.status}")
                if result.error_msg:
                    print(f"    错误: {result.error_msg}")


_TOKENIZER_PATH = os.environ.get(
    "PERCEVAL_TOKENIZER_PATH",
    os.environ.get("PERCEVAL_MODEL_PATH", "Qwen/Qwen2.5-VL-3B-Instruct"),
)
position_finder = SerialTokenPositionFinder(_TOKENIZER_PATH)


def encode_image(image_path=None, image_url=None):
    # 基于本地路径或者url获取图片的base64编码
    if image_path is not None:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif image_url is not None:
        response = requests.get(image_url)
        return base64.b64encode(response.content).decode('utf-8')
    else:
        raise ValueError("Either image_path or image_url must be provided.")


def process_message(each_data):
    text_parts = each_data['input_prompt'].split('<image>')
    content = []
    for i in range(len(each_data["images"])):
        if i < len(text_parts):
            if text_parts[i].strip():
                content.append({"type": "text", "text": text_parts[i].strip()})
        base64_image = encode_image(
            image_path=each_data["images"][i], 
            image_url=None
        )
        
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}})
            # 添加最后一段文本（如果存在）
    if len(text_parts) > len(each_data["images"]) and text_parts[-1].strip():
        content.append({"type": "text", "text": text_parts[-1].strip()})
    messages = [
        {"role": "system", "content": claude_tool_prompt},  # optional system prompt
        {"role": "user", "content": content}
        ]
    return messages


def extract_judgement_from_response(response_text):
    """
    从验证响应中提取answer部分并转换为列表
    
    Args:
        response_text (str): 包含<answer>标签的完整响应文本
        
    Returns:
        list: 如果有问题则返回问题列表，如果没有问题则返回空列表
    """
    # 使用字符串查找而不是正则表达式，提高性能
    start_tag = '<answer>'
    end_tag = '</answer>'
    
    start_idx = response_text.find(start_tag)
    if start_idx == -1:
        return []  # 如果没有找到answer标签，返回空列表
    
    start_idx += len(start_tag)
    end_idx = response_text.find(end_tag, start_idx)
    if end_idx == -1:
        return []  # 如果没有找到结束标签，返回空列表
    
    answer_content = response_text[start_idx:end_idx].strip()
    
    # 检查是否为"The response is correct"
    if "The response is correct" in answer_content:
        return []
    
    # 快速检查是否为列表格式
    if answer_content.startswith('[') and answer_content.endswith(']'):
        try:
            result = ast.literal_eval(answer_content)
            return result if isinstance(result, list) else []
        except (ValueError, SyntaxError):
            return []


def process_verify(predict_str: str, ground_truth: str, extra_info=None) -> float:
    
    question = extra_info['question']
    if '<image>' not in question:
        question = '<image>' + question
    input_prompt = claude_input_prompt.format(question=question, response=predict_str)
    send_data = {'input_prompt': input_prompt, 'images': extra_info['images']}
    messages = process_message(send_data)
    

    client_idx = random.randint(0, len(process_reward_client_list) - 1)
    client = process_reward_client_list[client_idx]
    model_name = process_reward_model_name_list[client_idx]

    # --- 开始重试循环 ---
    for attempt in range(MAX_RETRIES):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                seed=random.randint(0, 1000000),
                temperature=0,
                max_tokens=4096,
            )
            # 如果请求成功，直接跳出循环
            break
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            print(f" [WARNING ] 调用VLLM Server时发生可重试错误 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                # 指数退避 + 抖动
                # wait_time = (2 ** attempt) + random.uniform(0.1, 0.5)
                wait_time = 1
                print(f"           将在 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f" [ERROR ] 所有 {MAX_RETRIES} 次重试均失败。")
        except Exception as e:
            # 捕获其他不可预见的错误
            last_exception = e
            print(f" [ERROR ] 调用VLLM Server时发生未知错误: {e}")
            # 对于未知错误，通常不建议重试，直接中断
            break
    

    # --- 循环结束后的处理 ---
    # 检查是否成功获取了响应
    if chat_response is None or not chat_response.choices:
        print(f" [ERROR ] 未能从VLLM Server获取有效响应。最后一次错误: {last_exception}")
        return [], ""  # 返回默认的空值

    # chat_response = client.chat.completions.create(
    #     model=model_name,
    #     messages=messages,
    #     seed = random.randint(0, 1000000),
    #     temperature=0,
    #     max_tokens=4096,
    # )
    response = chat_response.choices[0].message.content
    # print(response)
    if not response:
        response = ''
    else:
        response = response.strip()

    judgement = extract_judgement_from_response(response)

    positions = position_finder.find_substring_positions(predict_str, judgement)

    new_positions = []
    for position in positions:
        region = [position.token_start, position.token_end]
        if -1 in region:
            continue
        new_positions.append(region)
    # print(new_positions)
    
    return new_positions, response

    





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
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

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


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'


    return full_prompt


def extract_answer(text):
    """
    从给定的文本中提取<answer></answer>标签内部的内容。
    
    参数:
        text (str): 包含<answer>标签的文本
        
    返回:
        str or None: 标签内部的内容，如果未找到则返回None。
    """
    # 使用非贪婪模式匹配<answer>和</answer>之间的内容
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


    


def compute_score(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    # pattern = re.compile(r'<\|im_start\|>assistant(.*?)$', re.DOTALL)  # 匹配最后一个 target 后的所有内容
    # match = pattern.search(predict_str)
    # if match:
    #     answer_text = match.group(1).strip()
    #     print(f'DEBUG{answer_text=}')
    # else:
    #     answer_text = ""

    question_text = extra_info['question']
    full_prompt = get_prompt(answer_text, ground_truth, question_text)


    client_idx = random.randint(0, len(judge_client_list) - 1)
    client = judge_client_list[client_idx]
    model_name = judge_model_name_list[client_idx]

    for attempt in range(MAX_RETRIES):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": full_prompt},
                ],
                seed = random.randint(0, 1000000),
                temperature=0.3,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except RETRYABLE_EXCEPTIONS as e:
            print(f"Attempt {attempt + 1} failed with a retryable error: {e}")
            print(full_prompt)
            if attempt < MAX_RETRIES - 1:
                continue
                # print(f"Waiting for {delay:.2f} seconds before retrying...")
                # time.sleep(delay)
                # delay *= BACKOFF_FACTOR # 增加下次等待的时间
            else:
                print(f"All {MAX_RETRIES} retries failed for this item. Skipping.")

        except Exception as e:
            # 捕获其他不可重试的意外错误（如代码错误、认证失败等）
            print(f"A non-retryable error occurred: {e}")
            print("Skipping this item without further retries.")
            break # 直接跳出重试循环，不再尝试
    # print(response)
    try:
        if 'Judgement:' in response:
            response = response.split('Judgement:')[-1].strip()
            if '1' in response:
                acc_reward = 1.0
            elif '0' in response:
                acc_reward = 0.0
            else:
                print(f' [WARNING] resp format error {response=}')
                acc_reward = 0.0
        else:
            if response == '1':
                acc_reward = 1.0
            elif response == '0':
                acc_reward = 0.0
            else:
                print(f' [WARNING] resp format error {response=}')
                acc_reward = 0.0
    except Exception as e:
        print(f' [WARNING] server has not returned {chat_response=}')
        acc_reward = 0.0
        is_format_error = True
    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    # reward 1
    # return 0.8 * acc_reward + 0.2 * format_reward + 0.4 * tool_reward_base
    # reward 2
    is_format_error = not check_format_by_regex_whitespace(predict_str)
    if (not is_format_error) and acc_reward > 0:
        final_reward = 1.0
    else:
        final_reward = 0.0
    # return 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward
    return [final_reward, acc_reward, format_reward]

    # reward 2 
    # return 1.0 * acc_reward + 0.2 * format_reward + 1.0 * tool_reward + 0.2 * tool_reward_base
    # reward 3
    # tool_reward_alpha = 1.2 if count_vision_1 > 0 else 0.0
    # return 1.0 * acc_reward * tool_reward_alpha + 0.2 * format_reward
    # reward 4
    # extra_reward = tool_reward_base * (count_vision_1 - 1) * (1 - acc_reward)
    # return  0.8 * acc_reward + 0.2 * format_reward + 0.4 * tool_reward_base  + 0.2 * extra_reward




def compute_common_reasoning(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = extract_answer(predict_no_think) # predict_no_think.split("<answer>")[-1].split("</answer>")[0].strip()
    if not answer_text:
        acc_reward = 0.0
        is_format_error = True
    elif len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True
    else:
        question_text = extra_info['question']
        client_idx = random.randint(0, len(judge_client_list) - 1)
        client = judge_client_list[client_idx]
        model_name = judge_model_name_list[client_idx]
        full_prompt = COMMON_VERIFY_PROMPT.format(
            query=question_text,
            gold_ans=ground_truth,
            pred_ans=answer_text,
        )

        acc_reward = 0.0
        for ix in range(8):
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed = random.randint(0, 1000000),
                temperature=0.5,
            )
            response = chat_response.choices[0].message.content.strip()
            judgement = response.split('## Equivalence Judgement')[-1].lower()
            if 'true' in judgement and 'false' not in judgement:
                acc_reward = 1.0
                break
            elif 'false' in judgement and 'true' not in judgement:
                acc_reward = 0.0
                break
            else:
                print(f' [ERROR] judgement format invalid: {judgement}')
                continue

    tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    is_format_error = not check_format_by_regex_whitespace(predict_str)
    if (not is_format_error) and acc_reward > 0:
        final_reward = 1.0
    else:
        final_reward = 0.0
    # print(f' [DEBUG] query={extra_info["question"]}, {ground_truth=}, {answer_text=}, {acc_reward=}, {format_reward=}')
    return [final_reward, acc_reward, format_reward]


def rule_math_verify(ground_truth, model_answer):
    gold = parse(ground_truth)
    answer = parse(model_answer)
    return verify(gold, answer)


def generative_verify(query, ground_truth, model_answer):
    client_idx = random.randint(0, len(judge_client_list) - 1)
    client = judge_client_list[client_idx]
    model_name = judge_model_name_list[client_idx]

    full_prompt = MATH_VERIFY_PROMPT.format(
        query=query,
        gold_ans=ground_truth,
        pred_ans=model_answer,
    )

    response = ""
    for it in range(8):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed = random.randint(0, 1000000),
                temperature=0.0,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f' [ERROR math] generative_verify error: {e}')
            continue
    
    judgement = response.split('## Equivalence Judgement')[-1].lower()
    if 'true' in judgement and 'false' not in judgement:
        return True
    elif 'false' in judgement and 'true' not in judgement:
        return False
    else:
        print(f' [ERROR math] verify bug output: ')


def compute_score_math(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    model_answer = ""
    predict_no_think = predict_str.split('</think>')[-1].strip()
    answer_pattern = r'\\boxed{([^}]+)}'
    answer_list = re.findall(answer_pattern, predict_no_think, flags=re.DOTALL)
    if len(answer_list) == 0:
        acc_reward = 0.0
        is_format_error = True
    else:
        if len(answer_list) > 1:
            is_format_error = True

        model_answer = answer_list[-1]
        if rule_math_verify(ground_truth, model_answer):
            acc_reward = 1.0
        else:
            acc_reward = 1.0 if generative_verify(extra_info['question'], ground_truth, model_answer) else 0.0
    
    format_reward = -1.0 if is_format_error else 0.0
    # print(f' [DEBUG] query={extra_info["question"]}, {ground_truth=}, {model_answer=}, {acc_reward=}, {format_reward=}')
    return [1.2 * acc_reward + 0.4 * format_reward, acc_reward, format_reward]



def _calculate_single_iou(interval1: list[int], interval2: list[int]) -> float:
    """计算两个单一区间 [start, end] 的交并比 (IoU)。"""
    start1, end1 = interval1
    start2, end2 = interval2

    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    
    intersection_length = max(0, intersection_end - intersection_start)

    if intersection_length == 0:
        return 0.0

    len1 = end1 - start1
    len2 = end2 - start2
    union_length = len1 + len2 - intersection_length
    
    if union_length == 0:
        return 1.0 if intersection_length > 0 else 0.0

    return intersection_length / union_length

def _merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """合并一个区间列表中的重叠或相邻区间。"""
    if not intervals:
        return []
    
    # 按起始位置排序
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        # 如果当前区间与上一个合并区间重叠或相邻，则合并
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
            
    return merged


def compute_interval_set_metrics(
    ground_truth_intervals: list[list[int]], 
    predicted_intervals: list[list[int]], 
    iou_threshold: float = 0.5
) -> dict:
    """
    计算两组token位置区间的重叠度指标（精确率, 召回率, F1, 宏观IoU）。

    Args:
        ground_truth_intervals (list[list[int]]): 真实问题区间的列表, e.g., [[10, 20], [35, 40]]。
        predicted_intervals (list[list[int]]): 模型预测的问题区间的列表。
        iou_threshold (float): 判断一个预测是否“命中”一个真实区间的IoU阈值。

    Returns:
        dict: 包含 'precision', 'recall', 'f1_score', 'macro_iou' 的字典。
    """
    # --- 1. 计算精确率、召回率和F1分数 ---
    
    true_positives = 0
    num_gt = len(ground_truth_intervals)
    num_pred = len(predicted_intervals)
    
    if num_gt == 0 and num_pred == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'macro_iou': 1.0}

    if num_pred == 0:
        # 没有做出任何预测，精确率未定义（或计为1），召回率为0
        return {'precision': 1.0, 'recall': 0.0, 'f1_score': 0.0, 'macro_iou': 0.0}
        
    if num_gt == 0:
        # 全是误报，召回率未定义（或计为1），精确率为0
        return {'precision': 0.0, 'recall': 1.0, 'f1_score': 0.0, 'macro_iou': 0.0}

    # 使用贪心策略进行匹配
    gt_matched = [False] * num_gt
    
    # 遍历每个预测区间
    for pred_interval in predicted_intervals:
        best_iou = 0
        best_gt_idx = -1
        # 找到与当前预测区间IoU最高的、且尚未匹配的真实区间
        for i, gt_interval in enumerate(ground_truth_intervals):
            iou = _calculate_single_iou(pred_interval, gt_interval)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        # 如果最佳匹配超过阈值，并且该真实区间未被其他预测匹配过
        if best_gt_idx != -1 and best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
            true_positives += 1
            gt_matched[best_gt_idx] = True # 标记为已匹配

    false_positives = num_pred - true_positives
    false_negatives = num_gt - true_positives

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    # --- 2. 计算宏观IoU (Macro IoU) ---
    
    # 分别合并两组区间
    merged_gt = _merge_intervals(ground_truth_intervals)
    merged_pred = _merge_intervals(predicted_intervals)
    
    # 计算合并后区间的总交集和总并集
    total_intersection = 0
    for gt_part in merged_gt:
        for pred_part in merged_pred:
            intersection_start = max(gt_part[0], pred_part[0])
            intersection_end = min(gt_part[1], pred_part[1])
            total_intersection += max(0, intersection_end - intersection_start)

    total_gt_len = sum(end - start for start, end in merged_gt)
    total_pred_len = sum(end - start for start, end in merged_pred)
    total_union = total_gt_len + total_pred_len - total_intersection

    macro_iou = total_intersection / total_union if total_union > 0 else 1.0 if total_intersection > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'macro_iou': macro_iou
    }


# (假设上面的 _calculate_single_iou, _merge_intervals, compute_interval_set_metrics 函数已定义)

# 假设 position_finder 已经像您原代码一样被初始化
# position_finder = SerialTokenPositionFinder('/path/to/your/tokenizer')

def evaluate_process_verification(
    predict_str: str, ground_truth: str, extra_info=None
) -> dict:
    """
    一个完整的流程函数，用于评估PRM的预测结果。

    Args:
        response_text (str): 模型的完整响应。
        ground_truth_phrases (list[str]): 真实的问题子句列表。
        predicted_phrases (list[str]): PRM预测的问题子句列表。
        position_finder (SerialTokenPositionFinder): token定位器。
        iou_threshold (float): IoU匹配阈值。

    Returns:
        dict: 包含评估指标的字典。
    """
    # 如果预测结果是"The response is correct"，则预测的问题列表为空
    # if predicted_phrases and "The response is correct" in predicted_phrases[0]:
    #     predicted_phrases = []
    is_format_correct = check_format_by_regex_whitespace(predict_str)
    predicted_phrases = extract_judgement_from_response(predict_str)
    ground_truth_phrases = extract_judgement_from_response(ground_truth)
    original_response_text = extra_info['original_response']

    # 1. 获取所有真实问题子句的token位置
    gt_positions = position_finder.find_substring_positions(original_response_text, ground_truth_phrases)
    ground_truth_intervals = []
    for pos in gt_positions:
        if pos.status == "success":
            ground_truth_intervals.append([pos.token_start, pos.token_end])

    # 2. 获取所有预测问题子句的token位置
    pred_positions = position_finder.find_substring_positions(original_response_text, predicted_phrases)
    predicted_intervals = []
    for pos in pred_positions:
        if pos.status == "success":
            predicted_intervals.append([pos.token_start, pos.token_end])
            
    # 3. 计算并返回指标
    iou_threshold = 0.5
    metrics = compute_interval_set_metrics(
        ground_truth_intervals,
        predicted_intervals,
        iou_threshold
    )
    final_reward = metrics['macro_iou'] if is_format_correct else 0.0
    format_reward = 1.0 if is_format_correct else 0.0
    acc_reward = metrics['macro_iou']
    res = [final_reward, acc_reward, format_reward]
    # 可以在这里添加一些调试信息
    # print("[DEBUG] GT Intervals:", ground_truth_intervals)
    # print("[DEBUG] Pred Intervals:", predicted_intervals)
    # print("[DEBUG] Metrics:", metrics)
    
    return res




def adaptive_comput_score(data_source, solution_str, ground_truth, extra_info=None, verify_process=False):
    critic_flag = False
    if ('vstar' in data_source) or (data_source in ['vstar', 'vl_agent', 'chart']):
        # from . import vl_agent 
        res = compute_score(solution_str, ground_truth, extra_info)
        

    elif data_source in ['geoguessr']:
        # from . import vl_agent
        res = compute_common_reasoning(solution_str, ground_truth, extra_info)

    elif ('ViRL' in data_source) or ('math' in data_source) or (data_source in ['thinklite_eureka', 'xince']):
        # from . import vl_agent
        # res = compute_score_math(solution_str, ground_truth, extra_info)
        res = compute_common_reasoning(solution_str, ground_truth, extra_info)
    elif 'critic' in data_source:
        critic_flag = True
        res = evaluate_process_verification(solution_str, ground_truth, extra_info)
    else:
        raise NotImplementedError

    final_res = {
        'score': res[0], 
        'acc_reward': res[1], 
        'format_reward': res[2], 
        'critic_flag': critic_flag
    }

    if (verify_process is True) and ('test' not in data_source):
        # print(f' [DEBUG] query={extra_info["question"]}, {data_source=}, {ground_truth=}, {solution_str=}, ')
        if 'vstar' in data_source:
            positions, verification = process_verify(solution_str, ground_truth, extra_info)
        else:
            positions = []
            verification = 'No verification.'
        final_res['positions'] = positions
        final_res['verification'] = verification


    if isinstance(final_res, dict):
        return final_res
    elif isinstance(final_res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])



if __name__ == '__main__':
    # predict_str = "The answer is <think> 2 + 2 = 4 </think> <answer> right </answer> <answer> left </answer>"
    # ground_truth = "left"
    # extra_info = {'answer': 'The woman is to the left of the man who is holding the camera.', 'id': 0, 'image': '/cpfs/user/honglingyi/DATA/LLM/Vstar/gqa/images/713270.jpg', 'pred_ans': 'The woman is to the right of the man who is holding the camera.', 'question': 'Is the woman to the left or to the right of the man who is holding the camera?'}

    # score = compute_score(predict_str, ground_truth, extra_info)
    # print(f"Score: {score}")
    # response = "<think>The user asks about the animals. I see a cat on the mat and a dog under the table.</think><answer>There is a cat and a dog.</answer>"
    # 这是被评估的原始文本
    original_response_text = "<think>The user asks about the animals. I see a cat on the mat and a dog under the table.</think><answer>There is a cat and a dog.</answer>"
    
    # 构建 extra_info，在所有测试用例中复用
    extra_info = {'original_response': original_response_text}

    # --- Test Case 1: Perfect Match ---
    # 预测和GT几乎一样
    print("--- Test Case 1: Perfect Match ---")
    gt_phrases1 = ['a dog under the table']
    pred_phrases1 = ['dog under the table']
    # 构建函数需要的字符串输入
    ground_truth_str1 = f"<answer>{str(gt_phrases1)}</answer>"
    predict_str1 = f"<answer>{str(pred_phrases1)}</answer>"
    metrics1 = evaluate_process_verification(predict_str1, ground_truth_str1, extra_info)
    print(f"Metrics: {metrics1}\n") # 预期: P, R, F1, IoU 都接近 1.0

    # --- Test Case 2: False Negative (漏报) ---
    # GT有两个错误，但模型只预测了一个
    print("--- Test Case 2: False Negative (漏报) ---")
    gt_phrases2 = ["cat on the mat", "dog under the table"]
    pred_phrases2 = ["dog under the table"]
    ground_truth_str2 = f"<answer>{str(gt_phrases2)}</answer>"
    predict_str2 = f"<answer>{str(pred_phrases2)}</answer>"
    metrics2 = evaluate_process_verification(predict_str2, ground_truth_str2, extra_info)
    print(f"Metrics: {metrics2}\n") # 预期: P=1.0, R=0.5, F1≈0.667

    # --- Test Case 3: False Positive (误报) ---
    # GT只有一个错误，但模型预测了两个
    print("--- Test Case 3: False Positive (误报) ---")
    gt_phrases3 = ["dog under the table"]
    pred_phrases3 = ["cat on the mat", "dog under the table"]
    ground_truth_str3 = f"<answer>{str(gt_phrases3)}</answer>"
    predict_str3 = f"<answer>{str(pred_phrases3)}</answer>"
    metrics3 = evaluate_process_verification(predict_str3, ground_truth_str3, extra_info)
    print(f"Metrics: {metrics3}\n") # 预期: P=0.5, R=1.0, F1≈0.667

    # --- Test Case 4: No errors predicted (Correctly) ---
    # GT没有错误，模型也正确地判断没有错误
    print("--- Test Case 4: No errors predicted (Correctly) ---")
    gt_phrases4 = []
    # 模型可以通过返回特定字符串或空列表来表示“无错误”
    predict_str4 = "<answer>The response is correct</answer>"
    ground_truth_str4 = f"<answer>{str(gt_phrases4)}</answer>"
    metrics4 = evaluate_process_verification(predict_str4, ground_truth_str4, extra_info)
    print(f"Metrics: {metrics4}\n") # 预期: P, R, F1, IoU 都为 1.0

    # --- Test Case 5: No errors predicted (Incorrectly, FN) ---
    # GT有错误，但模型错误地判断没有错误
    print("--- Test Case 5: No errors predicted (Incorrectly, FN) ---")
    gt_phrases5 = ["dog under the table"]
    predict_str5 = "<answer>The response is correct</answer>"
    ground_truth_str5 = f"<answer>{str(gt_phrases5)}</answer>"
    metrics5 = evaluate_process_verification(predict_str5, ground_truth_str5, extra_info)
    print(f"Metrics: {metrics5}\n") # 预期: P=1.0 (因为没预测), R=0.0, F1=0.0
    