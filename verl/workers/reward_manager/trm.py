# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict

import torch
import time # 导入 time 模块用于设置延迟

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

import datetime
import os
import concurrent
from zoneinfo import ZoneInfo

from filelock import FileLock
import json

# 1. 定义北京时区
beijing_tz = ZoneInfo("Asia/Shanghai") 

# 2. 获取当前时间并应用时区
#    datetime.now(tz) 是推荐的获取带时区时间的方式
beijing_now = datetime.datetime.now(beijing_tz)


MAX_RETRIES = 2      # 每个任务最多重试2次 (总共尝试 1 + 2 = 3次)
RETRY_DELAY = 5


@register("trm")
class TRMRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **reward_kwargs) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.reward_kwargs = reward_kwargs
        self.verify_process = self.reward_kwargs.get('verify_process', True)
        self.record_path = self.reward_kwargs.get('record_path', '')
        self.max_workers = self.reward_kwargs.get('max_workers', 128)
        #    datetime.now(tz) 是推荐的获取带时区时间的方式
        beijing_now = datetime.datetime.now(beijing_tz)

        # 格式化为文件名（只包含月份和日期）
        file_name = beijing_now.strftime("%m-%d_%H-%M-%S") + '.jsonl'
        self.record_name = os.path.join(self.record_path, file_name)
        # self.verify_process =
        print(f'verify_process-- type: {type(self.verify_process)}, value: {self.verify_process}')
        print(f'record_name-- type: {type(self.record_name)}, value: {self.record_name}')
        

    def pack_data(self, data_dict):
        data_item = data_dict

        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch[self.reward_fn_key]
        extra_info = data_item.non_tensor_batch.get("extra_info", None)
        
        packed_data = {
            "compute_args": {
                "data_source": data_source,
                "solution_str": response_str,
                "ground_truth": ground_truth,
                "extra_info": extra_info,
                "verify_process": self.verify_process,
            },
            "post_process_info": {
                "valid_response_length": valid_response_length.item(),
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "extra_info": extra_info,
            }
        }
        return packed_data



    def __call__(self, data: DataProto, return_dict=False, global_step=-1):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        tasks_to_process = []
        # ... (这部分代码与之前完全相同) ...
        for i in range(len(data)):
            # packed_data = self.pack_data(data[i])
            # packed_data['index'] = i
            # tasks_to_process.append(packed_data)
            packed_data = self.pack_data(data[i])
            packed_data['index'] = i
            # 为每个任务初始化重试计数器
            packed_data['retries'] = 0 
            tasks_to_process.append(packed_data)

        # =========================================================================
        # 2. 并行执行阶段: 使用 ThreadPoolExecutor 并行调用网络请求
        # =========================================================================
        results = [None] * len(tasks_to_process)
        
        # max_workers 的值需要小心设置，它代表最大并发请求数。
        # 如果目标服务器有速率限制，这个值不应设得太高。
        # 可以从 8 或 16 开始尝试，根据服务器的承受能力和网络状况进行调整。
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.compute_score, **task["compute_args"]): task
                for task in tasks_to_process
            }

            # 步骤 2: 使用 while 循环，只要还有未完成的 future，就一直处理
            while futures:
                # 等待至少一个 future 完成
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )

                # 步骤 3: 遍历所有已完成的 future
                for future in done:
                    # 从待办列表（futures 字典）中移除已完成的 future，并获取其对应的 task
                    task = futures.pop(future)
                    index = task["index"]
                    
                    try:
                        # 尝试获取结果
                        score = future.result()
                        # 成功！将结果存入 results 列表
                        results[index] = (score, task["post_process_info"])
                        
                    except Exception as exc:
                        # 出现异常，执行重试逻辑
                        print(f"Task for index {index} generated an exception: {exc}")
                        print(task["compute_args"])
                        if task['retries'] < MAX_RETRIES:
                            task['retries'] += 1
                            print(f"Retrying task for index {index} (Attempt {task['retries']}/{MAX_RETRIES}). Waiting for {RETRY_DELAY} seconds...")
                            time.sleep(RETRY_DELAY)
                            
                            # 重新提交任务
                            new_future = executor.submit(self.compute_score, **task["compute_args"])
                            
                            # **关键步骤**: 将新的 future 和同一个 task 重新加入到 futures 字典中
                            # 这样 while 循环下一次就会等待这个 new_future 的完成
                            futures[new_future] = task
                        else:
                            # Give up after MAX_RETRIES. Optionally dump the failed
                            # task to PERCEVAL_FAILED_TASK_LOG for offline inspection.
                            print(f"Task for index {index} failed after {MAX_RETRIES} retries. Giving up.")
                            results[index] = (None, task["post_process_info"])
                            _failed_log = os.environ.get("PERCEVAL_FAILED_TASK_LOG", "")
                            if _failed_log:
                                os.makedirs(os.path.dirname(_failed_log) or ".", exist_ok=True)
                                with FileLock(_failed_log + ".lock"):
                                    with open(_failed_log, "a") as f:
                                        f.write(json.dumps(task, ensure_ascii=False) + "\n")


            # as_completed 会在任何 future (无论是初始的还是重试的) 完成时返回它
            # for future in concurrent.futures.as_completed(future_to_task):
            #     task = future_to_task[future]
            #     index = task["index"]
                
            #     try:
            #         # 尝试获取结果
            #         score = future.result() 
            #         results[index] = (score, task["post_process_info"])
            #         # print(f"Task for index {index} completed successfully.") # 可以取消注释来调试

            #     except Exception as exc:
            #         # 关键的重试逻辑
            #         print(f"Task for index {index} generated an exception: {exc}")
                    
            #         if task['retries'] < MAX_RETRIES:
            #             task['retries'] += 1
            #             print(f"Retrying task for index {index} (Attempt {task['retries']}/{MAX_RETRIES}). Waiting for {RETRY_DELAY} seconds...")
                        
            #             # 等待一段时间再重试
            #             time.sleep(RETRY_DELAY)
                        
            #             # 重新提交任务到线程池
            #             new_future = executor.submit(self.compute_score, **task["compute_args"])
                        
            #             # **非常重要的一步**: 
            #             # 将新的 future 和旧的 task 关联起来，这样 as_completed 才能追踪到它
            #             future_to_task[new_future] = task 
            #         else:
            #             # 如果达到最大重试次数，则放弃
            #             print(f"Task for index {index} failed after {MAX_RETRIES} retries. Giving up.")
            #             # 即使失败，也用 None 占位，保持结果列表长度一致
            #             results[index] = (None, task["post_process_info"])
            # # for future in concurrent.futures.as_completed(future_to_task):
            #     task = future_to_task[future]
            #     index = task["index"]
            #     try:
            #         # 获取结果。如果 compute_score 内部没有处理超时，future.result() 可能会一直阻塞
            #         score = future.result() 
            #         results[index] = (score, task["post_process_info"])
            #     except Exception as exc:
            #         # 关键：处理网络请求可能发生的任何异常（超时、连接错误、服务器错误等）
            #         print(f"Task for index {index} generated an exception: {exc}")
            #         # 即使失败，也用 None 占位，保持结果列表长度一致
            #         results[index] = (None, task["post_process_info"])

        # =========================================================================
        # 3. 后处理阶段: 串行处理所有结果 (快速，无需并行)
        # =========================================================================
        for i, result in enumerate(results):
            if result is None or result[0] is None:
                print(f"Skipping result for index {i} due to a processing error.")
                continue

            score, info = result
            # print(f'---score---{score}')
            # print(f'---info---{info}')
            if 'verification' in score:
                info['verification'] = score.pop('verification')
            info['score'] = score["score"]
            if 'critic_flag' in score:
                info['critic_flag'] = score.pop('critic_flag')
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            # info['reward'] = reward.item()
            # print(f'---score---{score}')
            # print(f'---info---{info}')
             
            reward_tensor[i, info["valid_response_length"] - 1] = reward
            info.update({'step': global_step})

            # data_source = info["data_source"]
            self.write_record(info)

        if self.verify_process is True and 'positions' in reward_extra_info:
            positions = reward_extra_info['positions']
            padded_positions = pad_nested_list(positions, padding_value=-100)
            reward_extra_info['positions'] = padded_positions


        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


    def write_record(self, data):
        with FileLock(self.record_name + ".lock"):
            with open(self.record_name, "a") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")



def pad_nested_list(data, padding_value=0):
    """
    递归地将一个不规整的嵌套列表填充成一个规整的矩形结构。

    Args:
        data (list): 输入的不规整嵌套列表。
        padding_value (any): 用于填充的数值。对于深层嵌套，
                             会自动构建与层级匹配的填充结构。

    Returns:
        list: 填充后变得规整的嵌套列表。
    """
    # 步骤 1: 递归地获取目标形状
    def get_target_shape(lst, shape, level=0):
        if not isinstance(lst, list):
            return
        
        # 确保shape列表足够长以记录当前层级的维度
        while level >= len(shape):
            shape.append(0)
        
        # 更新当前层级的最大长度
        shape[level] = max(shape[level], len(lst))
        
        # 递归到下一层级
        for item in lst:
            get_target_shape(item, shape, level + 1)
    
    target_shape = []
    get_target_shape(data, target_shape)

    # 步骤 2: 根据目标形状，递归地创建填充用的“模板”
    def create_padding_element(shape, value):
        if not shape:  # 如果没有更多维度，直接返回值
            return value
        # 否则，根据当前维度创建列表，并递归创建子元素
        return [create_padding_element(shape[1:], value)] * shape[0]

    # 步骤 3: 递归地填充原始数据
    def pad_to_shape(lst, shape):
        if not isinstance(lst, list):
            # 如果当前数据不是列表（比如是单个数字），但仍有维度需要填充
            # 这通常意味着原始数据结构在这里就结束了，我们需要根据剩余形状创建填充
            return create_padding_element(shape, lst) # 将当前值作为最终填充值

        if not shape: # 如果没有更多维度，直接返回当前列表（或元素）
            return lst
            
        current_dim = shape[0]
        child_shape = shape[1:]
        
        # 首先，递归地填充所有子元素
        padded_children = [pad_to_shape(item, child_shape) for item in lst]
        
        # 然后，填充当前列表本身
        num_to_pad = current_dim - len(padded_children)
        if num_to_pad > 0:
            # 创建一个符合子元素目标形状的填充模板
            padding_element = create_padding_element(child_shape, padding_value)
            padded_children.extend([padding_element] * num_to_pad)
        
        return padded_children

    return pad_to_shape(data, target_shape)