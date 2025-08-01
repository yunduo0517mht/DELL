import os
import random
import json

import pandas as pd
import torch
from openai import OpenAI
from requests.packages import target
from tqdm import tqdm
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt_tab')

#  load LLMs, we employ mistral-7b as LLM, from https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 5, 7'
molde = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-fea22c819b7b4be69fb3b821a7e804e6",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

def construct_length(text):
    sents = nltk.sent_tokenize(text)
    out = ''
    for sent in sents:
        if len(out) + len(sent) + 1 <= 640:
            out = out + ' ' + sent
        else:
            break
    return out


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


characters = {
    'gender': ['male', 'female'],
    'age': ['0-17', '18-29', '30-49', '50-64', '65+'],
    'ethnicity': ['White', 'Black', 'Hispanic'],
    'education': ['college grad', 'some college', 'HS or less'],
    'income': ['75000 or more', '30000-74999', 'less than 30000'],
    'party': ['Republican', 'Democrat'],
    'vote': ['registered to vote', 'probably to vote', 'not registered']
}


profile_text = {
    'gender': ['You are male.', 'You are female.'],
    'age': ['You are under 17 years old.', 'You are 18 to 29 years old.', 'You are 30 to 49 years old.',
            'You are 50 to 64 years old.', 'You are over 65 years old.'],
    'ethnicity': ['Racially, you are {}.'.format(_) for _ in ['White', 'Black', 'Hispanic']],
    'income': ['Financially, your annual family income is {}.'.format(_)
               for _ in ['more than 75,000', '30,000 to 74,999', 'less than 30,000']],
    'education': ['Educationally, you {}.'.format(_)
                  for _ in ['are a college grad', 'haven\'t graduated from college',
                            'have a high school diploma or less']],
    'party': ['Politically, you are a {}.'.format(_)
              for _ in ['Democrat', 'Republican']],
    'vote': ['Meanwhile, you are {}.'.format(_)
             for _ in ['registered to vote', 'probably registered to vote', 'not registered to vote']]
}


def generate_a_character():
    profile = 'You are a social media user. '
    for item in profile_text:
        profile += random.choice(profile_text[item]) + ' '
    return profile


def get_news_comment(user, news):
    prompt = '{}\nYou view a piece of news with the following content.\n'.format(user)
    prompt += 'News:\n{}\n'.format(news)
    prompt += 'Task:\nClassify news from "information sharing," "socialization," "political propaganda," "interest propaganda," "creating chaos or public opinion storms," "emotional venting," "rumor dissemination," and "information dissemination," and provide probabilities. Your comment is limited to 50 words. '
    prompt += '\nCategory：(类别，概率),comment：()'
    res = get_reply(prompt)
    return res


def get_comment_comment(user, news, comments):
    prompt = '{}\nYou view a piece of news and a related comment chain on social media, ' \
             'and their contents are as follows.\n'.format(user)
    prompt += 'News:\n{}\n'.format(news)
    for index, comment in enumerate(comments):
        prompt += 'Comment {}: {}\n'.format(index+1, comment)
    prompt += 'Task:\nClassify news from "information sharing," "socialization," "political propaganda," "interest propaganda," "creating chaos or public opinion storms," "emotional venting," "rumor dissemination," and "information dissemination," and provide probabilities.\n  '.format(len(comments))
    prompt += 'Your reply is limited to 50 words. '
    prompt += '\nCategory：(类别，概率),comment：()'
    res = get_reply(prompt)
    return res


def get_comment_comment_chain(user, news, comment_chains):
    prompt = '{}\nYou view a piece of news and related comment chains on social media, ' \
             'and their contents are as follows.\n'.format(user)
    prompt += 'News:\n{}\n'.format(news)
    for chain_index, comments in enumerate(comment_chains):
        prompt += 'Comment Chain {}:\n'.format(chain_index+1)
        for index, comment in enumerate(comments):
            prompt += 'Comment {}: {}\n'.format(index+1, comment)
    prompt += 'Task:\nClassify news from "information sharing," "socialization," "political propaganda," "interest propaganda," "creating chaos or public opinion storms," "emotional venting," "rumor dissemination," and "information dissemination," and provide probabilities.and select a comment chain that you would most like to comment. ' \
              'The answer format is: Category：(类别，概率),comment：()\n'
    prompt += 'Answer:'
    res = get_reply(prompt)
    return res


def generate(news):
    def get_chain(node_id):
        chain_id = [node_id]
        while True:
            node_id = father[node_id]
            if node_id == 0:
                break
            chain_id.append(node_id)
        chain_id = list(reversed(chain_id))
        return [res[_] for _ in chain_id]

    def comment_comment(func_candidate):
        if len(func_candidate) == 1:
            comment = get_chain(func_candidate[0])
            res.append(get_comment_comment(user, res[0], comment))
            return func_candidate[0], None
        else:
            chain = [get_chain(_) for _ in func_candidate]
            func_reason = get_comment_comment_chain(user, res[0], chain)
            try:
                match = re.search(r'\d+', func_reason)
                choice_id = int(match.group(0)) - 1
                choice = chain[choice_id]
            except Exception as e:
                choice_id = random.choice(range(len(chain)))
                func_reason = str(e)
                choice = chain[choice_id]
            res.append(get_comment_comment(user, res[0], choice))
            return func_candidate[choice_id], func_reason

    def TF_IDF(res):
        source = []
        target = []
        # 使用 TfidfVectorizer 将评论转换为 TF-IDF 矩阵
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(res)

        # 计算余弦相似性
        cosine_sim = cosine_similarity(tfidf_matrix)

        # 记录相似性超过0.5的评论对
        similar_comments = []

        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                if cosine_sim[i][j] > 0.5:
                    source.append(i)
                    target.append(j)
        return source,target

    def extract_category_content(text):
        # 定义正则表达式：匹配 Category 到 Comment/comment 之间的内容（不区分大小写，允许换行符）
        pattern = re.compile(
            r'Category:\s*(.*?)\s*(?=Comment:|comment:)',  # 核心匹配逻辑
            re.IGNORECASE | re.DOTALL  # 忽略大小写、允许.匹配换行符
        )
        match = pattern.search(text)
        if not match:
            return None

        content = match.group(1).strip()
        # 清理末尾的冗余符号（如逗号、句号等）
        content = re.sub(r'[^\w\s%()-]+$', '', content).strip()

        # 解析内容，提取类别和概率
        categories = re.findall(r'([\w\s]+)\((\d+)%\)', content)

        # 将类别和概率转换为列表，并过滤掉概率小于等于20%的
        filtered_categories = [(cat.strip(), int(prob)) for cat, prob in categories if int(prob) > 20]

        # 如果没有符合条件的类别，返回None
        if not filtered_categories:
            return None

            # 找到概率最大的类别并剔除
        max_prob_category = max(filtered_categories, key=lambda x: x[1])
        filtered_categories.remove(max_prob_category)

        # 将剩余的类别重新组合成字符串
        result = ', '.join([f"{cat}" for cat, prob in filtered_categories])

        return result

    def find_common_word_indices(lst):
        # 创建一个空列表来存储有相同词的下标对
        source = []
        target = []
        lst = lst[1:]
        # 遍历列表中的每一对元素
        k = 0
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                # 获取当前对元素的值
                item1 = lst[i]
                item2 = lst[j]
                # 检查是否为非空字符串
                if item1 and item2 and isinstance(item1, str) and isinstance(item2, str):
                    # 检查是否有相同的词
                    if k < 6:
                        if item1 and item2 and (item1 in item2 or item2 in item1):
                            # 如果相互包含，保留下标对
                            source.append(i)
                            target.append(j)
                            k += 1

        return source, target
    torch.cuda.empty_cache()

    res = [news]
    users = []
    reasons = []

    size_limit = 11  # parameter to control the network size
    rt_ratio = 0.6 # parameter to control the probability of commenting on the news article, namely alpha in our paper
    height_ratio = 0.6  # parameter to balance the tree height and width, namely bata in our paper
    eps = 1e-5
    k = 3  # parameter to control candidate set size

    father = [None]
    height = [0]
    width = [0]
    source= []
    target = []
    source1 = []
    target1 = []
    graph_size = 1
    rt = 0

    while graph_size < size_limit:
        user = generate_a_character()
        users.append(user)
        if graph_size == 1:
            graph_size += 1
            father.append(rt)
            height.append(height[rt] + 1)
            width.append(0)
            res.append(get_news_comment(user, res[rt]))
            width[rt] += 1
            reasons.append(None)
        else:
            identify = random.uniform(0, 1)
            if identify <= rt_ratio:
                graph_size += 1
                father.append(rt)
                height.append(height[rt] + 1)
                width.append(0)
                res.append(get_news_comment(user, res[rt]))
                width[rt] += 1
                reasons.append(None)
            else:
                candidate = [_ for _ in range(1, graph_size)]
                probabilities = [height[_] * height_ratio + width[_] * (1 - height_ratio) + eps
                                 for _ in candidate]
                probabilities = np.array(probabilities)
                probabilities /= np.sum(probabilities)
                candidate_comment = np.random.choice(candidate, p=probabilities,
                                                     size=min(k, len(candidate)), replace=False)
                fa, reason = comment_comment(candidate_comment)
                reasons.append(reason)

                graph_size += 1
                father.append(fa)
                height.append(height[fa] + 1)
                width.append(0)
                width[fa] += 1
    father = [int(_) if _ is not None else None for _ in father]
    source, target = TF_IDF(res)[0],TF_IDF(res)[1]
    results = [extract_category_content(text) for text in res]
    source1, target1 = find_common_word_indices(results)[0],find_common_word_indices(results)[1]
    source=source+source1
    target=target + target1

    return {
        'father': father,
        'res': res,  # the comment content of each node
        'source':source,
        'target':target
    }


def main():
    random.seed(20250101)
    dataset_names = {
        'TASK1': ['weibo21'],
    }
    for task in dataset_names:
        for dataset in dataset_names[task]:
            dataset_name = dataset.replace('.json', '')
            if not os.path.exists('../data/networks'):
                os.mkdir('../data/networks')
            save_dir = f'../data/networks/{task}_{dataset}'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            data = json.load(open(f'../data/datasets/{task}/{dataset}.json', encoding='UTF-8'))
            #  load datasets
            for index, item in enumerate(tqdm(data, desc='{}, {}'.format(task, dataset_name), leave=False)):
                save_path = f'{save_dir}/{index}.json'
                if os.path.exists(save_path):
                    continue
                #  a way to limit the length of news articles, optiona
                in_text = construct_length(item[0])
                #  obtain the user-news network. If you want to generate user-news 原始网络 on other datasets,
                #  you can use the generate function.
                out = generate(in_text)  # the input is news article
                with open(save_path, 'w',encoding='utf-8') as f:
                    json.dump(out, f)
if __name__ == '__main__':
    main()
