import json
import os
import torch
from utils import get_reply
from tqdm import tqdm
from sklearn.metrics import f1_score
import re
from ensemble_utils import likelihood_weight
import numpy as np


dataset_names = {
    'TASK1': ['Pheme'],
}

#这个函数`get_metric`用于计算F1分数。它接受两个参数：`y_true`（真实标签）和`y_pred`（预测标签）。函数返回两个F1分数，分别是宏观平均（macro）和微观平均（micro），并将结果乘以100，以方便查看百分比。
def get_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro') * 100, \
        f1_score(y_true, y_pred, average='micro') * 100


def run(task, dataset):
    if not os.path.exists('results'):
        os.mkdir('results')
    save_dir = f'results/confidence_{task}_{dataset}.json'
    data = json.load(open(f'../data/datasets/{task}/{dataset}.json'))
    test_indices = json.load(open(f'../../DELL-main/data/datasets/split/{task}_{dataset}/test.json' ))

    # experts and description of each expert
    text_augmentation = ['None', 'emotion', 'framing', 'propaganda', 'retrieval', 'None', 'None']
    graph_augmentation = ['None', 'None', 'None', 'None', 'None', 'stance', 'relation']

    expert_descriptions = [
        'This expert is comprehensive. ',
        'This expert focuses on the emotion of this news. ',
        'This expert focuses on the framing of this news. ',
        'This expert focuses on the propaganda technology of this news. ',
        'This expert focuses on the external knowledge of this news. ',
        'This expert focuses on the stance of related comments on this news. ',
        'This expert focuses on the relation of related comments on this news. '
    ]

    all_experts = []
    for text, graph in zip(text_augmentation, graph_augmentation):
        # load prediction of each expert
        expert = torch.load(f'../expert/4.11(86)results/{task}_{dataset}_{text}_{graph}_test.pt')
        all_experts.append(expert)
    if os.path.exists(save_dir):
        out = json.load(open(save_dir))
    else:
        out = []
    for _, index in enumerate(tqdm(test_indices[len(out):])):
        news = data[index][0].strip()
        # truncate news article if needed
        prompt = 'News:\n{}\n\n'.format(news)
        prompt += 'Some experts give predictions and confidence scores about the news. '
        prompt += 'The higher the score, the more confidence the result is.\n'
        for _index in range(len(all_experts)):
            expert = all_experts[_index]
            expert_ans = expert[1][_]
            expert_ans = expert_ans.data.numpy()

            if task != 'TASK1':
                expert_likelihood = expert[0][_]
                expert_likelihood = torch.abs(expert_likelihood).data.numpy()
            else:
                expert_likelihood = expert[0][_]
                expert_likelihood = torch.softmax(expert_likelihood, dim=-1)
                expert_likelihood = torch.max(expert_likelihood, dim=-1)[0]
                expert_likelihood = torch.abs(expert_likelihood).data.numpy()
            expert_likelihood = np.round(expert_likelihood, 2)

            expert_info = 'Expert {}:\n'.format(_index + 1)
            expert_info += expert_descriptions[_index]

            expert_info += f'The expert predicts the label of this news is {expert_ans}. '
            if task == 'TASK1':
                expert_info += 'The confidence scores are {:.2f}.\n'.format(expert_likelihood)
            else:
                expert_info += 'The confidence scores are {}.\n'.format(expert_likelihood)
            prompt += expert_info
        prompt += '\n'
        prompt += 'Question:\nBased on the analysis of experts, please judge the final label of this news. '
        prompt += 'Give the label in the form of \"[your answer]\", do not give any explanation.\n'
        prompt += 'Label:'
        res = get_reply(prompt)
        out.append(res)
        json.dump(out, open(save_dir, 'w'))


def find_bracketed_substrings(input_string):
    pattern = r'\[(.*?)\]'
    res = re.findall(pattern, input_string)
    return res[0]

#这个代码的功能是评估一个任务的结果。它从文件中加载数据和预测结果，然后根据不同的条件生成预测列表。
# 在处理每个结果时，如果是特定条件下的任务，它会通过提取子字符串来获取预测。如果出现错误，就使用默认预测。
# 最后，它会输出一个度量指标来评价这些预测和真实标签之间的差异。
def evaluate(task, dataset):
    data = torch.load('../expert/4.11(86)results/{}_{}_None_None_test.pt'.format(task, dataset))
    label = data[-1].numpy().tolist()
    results = json.load(open(f'results/confidence_{task}_{dataset}.json'))
    default_preds = likelihood_weight(task, dataset).tolist()
    preds = []
    for index, item in enumerate(results):
        if isinstance(item, list):
            item = item[0]
        if task != 'TASK1':
            try:
                pred = find_bracketed_substrings(item)
                pred = [int(_) for _ in pred.split()]
                pred = [1 if _ != 0 else 0 for _ in pred]
                assert len(pred) == len(label[0])
            except Exception:
                pred = default_preds[index]
            preds.append(pred)
        else:
            if '1' in item:
                preds.append(1)
            elif '0' in item:
                preds.append(0)
            else:
                preds.append(default_preds[index])
    print(get_metric(label, preds))


def main():
    for task in dataset_names:
        for dataset in dataset_names[task]:

            #整个过程的目的是通过多个专家的分析来判断和记录新闻的标签。
            # 它加载任务和数据集的相关文件，包括新闻数据和测试索引。定义了一些专家的文本和图形增强方式，并且为每个专家提供了描述信息。在循环中，它加载每个专家的预测结果，并储存到一个列表中。
            #并添加到提示中。最后，构建一个问题并调用一个名为`get_reply`的函数来获取最终的标签，并将结果保存到文件中。
            run(task, dataset)

            #用于评估模型的预测表现。它做了以下几件事：1. 从指定路径加载数据集。2. 获取标签数据并转换为列表格式。3. 从结果文件中加载预测结果。
            #4. 使用一个默认预测函数生成默认预测值。5. 根据结果逐个处理每个项目，进行预测并存储在 `preds` 列表中。
            # 6. 针对不同的任务类型（任务1和其他任务），代码有不同的预测方式。
            # 7. 最后，通过调用 `get_metric` 函数计算并打印出标签和预测之间的衡量指标。
            evaluate(task, dataset)

#告诉大模型，置信度越高，月可信，要大模型自己生成
if __name__ == '__main__':
    main()
