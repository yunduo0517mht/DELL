import os
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


#  load LLMs, we employ mistral-7b as LLM, from https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
#  you can change it to other LLMs such as ChatGPT or Llama
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
# model_path = '/root/dataroot/models/mistralai/Mistral-7B-Instruct-v0.1'
# model = AutoModelForCausalLM.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
molde = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-fea22c819b7b4be69fb3b821a7e804e6",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

@torch.no_grad()
def get_reply(content, temperature=0.3):
    completion = molde.chat.completions.create(
        model="qwen2-72b-instruct",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': content}
        ]
    )
    return completion.choices[0].message.content

