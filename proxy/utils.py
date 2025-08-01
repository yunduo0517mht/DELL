import os

from openai import OpenAI
import torch
import nltk


#  load LLMs, we employ mistral-7b as LLM, from https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# model_path = 'D:\模型\huggingface_cache\hub\models--mistralai--Mistral-7B-Instruct-v0.1\snapshots/2dcff66eac0c01dc50e4c41eea959968232187fe'
# model = AutoModelForCausalLM.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
client = OpenAI(
        api_key="sk-fea22c819b7b4be69fb3b821a7e804e6",  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)
#     return decoded
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


def construct_length(text, length=10000):
    if len(text) < length:
        return text
    sents = nltk.sent_tokenize(text)
    out = ''
    for sent in sents:
        if len(out) + len(sent) + 1 <= length:
            out = out + ' ' + sent
        else:
            break
    return out

def net_intent_detection(net):
    item = generate_sequence_description(net)
    prompt = f"""  
            你是一位传播意图检测专家。请阅读以下新闻评论{item}
            请回答以下问题：  
            1. 这篇新闻的传播意图是否是为了传播虚假信息，概率是多少。  
            2. 这篇新闻的传播意图是否是为了煽动用户的情绪，概率是多少。  
            3. 这篇新闻的传播意图是否是为了从中获得利益，概率是多少。  
            4. 这篇新闻的传播意图是否是为了普及新闻，概率是多少。  

            用简明的方式回答，误导性：概率，煽动型：概率，利益驱动型：概率，普及型，概率,并说明理由，50字以内。
            """
    reply = get_reply(prompt)
    return reply

def generate_sequence_description(data):
    father_list = data["father"]
    # users_list = data["users"]
    res_list = data["res"]
    sequence_descriptions = []

    for i in range(1, len(father_list)):
        father_index = father_list[i]
        if father_index is  None:
            continue
        if father_index ==0:
            sequence_descriptions.append(f"用户{i}:\n评论了新闻：{res_list[i]}")
        else:
            sequence_descriptions.append(f"用户{i}:\n评论了用户{father_index}评论为：{res_list[i]}")
    sequence_descriptions= " ".join(sequence_descriptions)
    return sequence_descriptions

def text_intent_detection(news):
    res = [news]
    users = []
    for expert_role in expert_roles:
        users += expert_role
        comment = get_news_comment(news, expert_role)
        res.append(comment)  # 将评论添加到结果中
    res = " ".join(res)
    return get_content_intent(res)

expert_roles = [
    "新闻评议员, 擅长分析新闻报道背后的政治意图。",
    "心理学专家, 擅长分析新闻如何影响受众的心理和行为。",
    "社会学专家, 擅长分析新闻对社会舆论的影响。",
    "媒体伦理专家, 擅长评估新闻的伦理道德。"
]
def get_news_comment( news_content, expert_role):
    """
    让 LLM 以特定身份对新闻进行评论，并揣测新闻意图。
    """

    # 构建提示词，询问新闻意图
    prompt = f"""  
    你是一位{expert_role}。请阅读以下新闻报道，并从信念、欲望和计划三个方面揣测新闻的意图：  

    新闻报道：  
    {news_content}  

    请回答以下问题：  
    1. 这篇新闻的主要目的是否是为了说服读者相信某个观点（信念）？如果是，请说明是什么观点。  
    2. 这篇新闻是否试图激发读者对某件事物的渴望或厌恶（欲望）？如果是，请说明是什么事物和情感。  
    3. 这篇新闻是否试图影响读者的行为或让他们采取某种行动（计划）？如果是，请说明是什么行动。  

    请用简洁明了的语言回答。并且以{expert_role}：开头
    """
    #profile = generate_a_character()
    #prompt = profile + prompt
    # 获取 LLM 的回复
    reply = get_reply(prompt)

    return reply

def get_content_intent(res):
    """
        让 LLM 以特定身份对新闻进行评论，并揣测新闻意图。
        """

    # 构建提示词，询问新闻意图
    prompt = f"""  
        通过阅读专家们的评论{res}，进行一个总结，信念：旨在改变特定观点的认知（概率），欲望：旨在激发受众的渴望或厌恶（概率），计划：旨在影响受众的行为（概率）
        用简短的文字回答。
        """
    reply = get_reply(prompt)

    return reply

def intent_detection(spr_intent,text_intnet):
    prompt = f"""  
            你是一位意图检测专家。意图由传播意图和文本意图构成。
            请根据传播意图：{spr_intent}和文本意图：{text_intnet}分析，
            请简单总结：限制在50个字
            """
    # profile = generate_a_character()
    # prompt = profile + prompt
    # 获取 LLM 的回复
    reply = get_reply(prompt)

    return reply

# def rep_generation(text):
#     response = client.embeddings.create(
#             model="text-embedding-v3",
#             input=[text],  # 每次处理一个文本
#             dimensions=1024,
#             encoding_format="float"
#         )
#     embedding_vector = response.data[0].embedding  # 获取嵌入向量
#     tensor = torch.tensor(embedding_vector, dtype=torch.float32)  # 转换为 PyTorch 张量
#     return tensor

def rep_generation( text_list):
    reps=[]
    client = OpenAI(
            api_key="sk-fea22c819b7b4be69fb3b821a7e804e6",  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
        )

        # 使用迭代器循环处理每个文本
    for text in text_list:
        response = client.embeddings.create(
                model="text-embedding-v3",
                input=[text],  # 每次处理一个文本
                dimensions=1024,
                encoding_format="float"
            )
        embedding_vector = response.data[0].embedding  # 获取嵌入向量
        tensor = torch.tensor(embedding_vector, dtype=torch.float32)  # 转换为 PyTorch 张量
        reps.append(tensor)  # 将张量添加到结果列表中
    reps=torch.stack(reps)
    return reps

def framing_detection(text):
    prompt = f'News:\n{text}\nTask:\n'
    prompt += 'Framing is a strategic device and a central concept in political communication ' \
              'for representing different salient aspects and perspectives to convey the latent meaning of an issue. '
    prompt += 'Which framings does the news contain? Please choose the five most likely ones: '
    prompt += 'Economic; Capacity and resources; Morality; Fairness and equality; ' \
              'Legality, constitutionality and jurisprudence; Policy prescription and evaluation; ' \
              'Crime and punishment; Security and defense; Health and safety; Quality of life; Cultural identity;' \
              ' Among public opinion; Political; External regulation and reputation. '
    prompt += 'Please provide your reasoning.'
    prompt += '\nAnswer:'
    res = get_reply(prompt)
    return res

def sentiment_detection(text):
    prompt = f'News:\n{text}\nQuestion:\n'
    prompt += 'Which emotions does the news contain? ' \
              'Please choose the three most likely ones: anger, disgust, fear, happiness, sadness, and surprise. '
    prompt += 'Please provide your reasoning.'
    prompt += '\nAnswer:'
    res = get_reply(prompt)
    return res


def propaganda_detection(text):
    prompt = f'News:\n{text}\nTask:\n'
    prompt += 'Propaganda tactics are methods used in propaganda to convince an audience ' \
              'to believe what the propagandist wants them to believe. '
    prompt += 'Which propaganda techniques does the news contain? Please choose the five most likely ones: '
    prompt += 'Conversation Killer; Whataboutism; Doubt; Straw Man; Red Herring; Loaded Language; ' \
              'Appeal to Fear-Prejudice; Guilt by Association; Flag Waving; False Dilemma-No Choice; ' \
              'Repetition; Appeal to Popularity; Appeal to Authority; Name Calling-Labeling; Slogans; ' \
              'Appeal to Hypocrisy; Exaggeration-Minimisation; Obfuscation-Vagueness-Confusion; ' \
              'Causal Oversimplification. '
    prompt += 'Please provide your reasoning.'
    prompt += '\nAnswer:'
    res = get_reply(prompt)
    return res


def find_entity(text):
    prompt = f'News:\n{text}\n'
    prompt += 'TASK:\n'
    prompt += 'Identify five named entities within the news above that necessitate elucidation ' \
              'for the populace to understand the news comprehensively. '
    prompt += 'Ensure a diverse selection of the entities. The answer should in the form of python list.'
    prompt += '\nAnswer:'
    res = get_reply(prompt)
    return res


def get_stance(textA, textB):
    textA = construct_length(textA, length=5000)
    textB = construct_length(textB, length=5000)
    prompt = 'TASK:\nDetermine the stance of sentence 2 on sentence 1. ' \
             'Is it supportive, neutral or opposed? Provide your reasoning.\n'
    prompt += f'Sentence 1: {textA}\n'
    prompt += f'Sentence 2: {textB}\n'
    prompt += 'Answer:'
    res = get_reply(prompt)
    return res


def get_response(textA, textB):
    textA = construct_length(textA, length=5000)
    textB = construct_length(textB, length=5000)
    prompt = f'Sentence 1:\n{textA}\n'
    prompt += f'Sentence 2:\n{textB}\n'
    prompt += 'TASK:\n'
    prompt += 'Sentence 1 and Sentence 2 are two posts on social 原始网络. '
    prompt += 'Please judge whether the sentence 2 replies to the sentence 1. '
    prompt += 'Answer yes or no and provide the reasoning.\n'
    prompt += 'Answer:'
    res = get_reply(prompt)
    return res
