import json
import os.path
import torch
from utils import get_reply
from tqdm import tqdm
from sklearn.metrics import f1_score
import re


dataset_names = {
    'TASK1': ['Pheme'],

}


def get_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro') * 100, \
        f1_score(y_true, y_pred, average='micro') * 100


def run(task, dataset):
    if not os.path.exists('results'):
        os.mkdir('results')
    save_dir = f'results/selective_{task}_{dataset}.json'
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
        expert = torch.load('../expert/4.11(86)results/{}_{}_{}_{}_test.pt'.format(task, dataset, text, graph))
        all_experts.append(expert)
    if os.path.exists(save_dir):
        out = json.load(open(save_dir))
    else:
        out = []
    for _, index in enumerate(tqdm(test_indices[len(out):])):
        news = data[index][0]
        # truncate news article if needed
        prompt = 'News:\n{}\n'.format(news)
        for _index in range(len(all_experts)):
            expert_info = 'Expert {}: '.format(_index + 1)
            expert_info += expert_descriptions[_index] + '\n'
            prompt += expert_info
        prompt += 'To understand this news, which expert knowledge do you need?'
        prompt += ' Return a Python list, e.g. [expert 1, expert 2, expert 6].'
        #少样本
        res = get_reply(prompt)
        out.append(res)
        json.dump(out, open(save_dir, 'w'))


def find_bracketed_substrings(input_string):
    pattern = r'\[(.*?)\]'
    res = re.findall(pattern, input_string)
    return res[0]


def evaluate(task, dataset):
    text_augmentation = ['None', 'emotion', 'framing', 'propaganda', 'retrieval', 'None', 'None']
    graph_augmentation = ['None', 'None', 'None', 'None', 'None', 'stance', 'relation']
    all_expert = []
    for text, graph in zip(text_augmentation, graph_augmentation):
        data = torch.load(f'../expert/4.11(86)results/{task}_{dataset}_{text}_{graph}_test.pt')
        all_expert.append(data)
    # test_indices = json.load(open('split/{}_{}/test.json'.format(task, dataset)))
    expert_info = json.load(open(f'results/selective_{task}_{dataset}.json'))
    all_preds = []
    for _index, expert_data in enumerate(expert_info):
        expert = []
        preds = []
        weight = []
        for _ in range(1, 8):
            if str(_) in expert_data:
                expert.append(_ - 1)
        if len(expert) == 0:
            expert = [0, 1, 2, 3, 4, 5, 6]
        for expert_id in expert:
            data = all_expert[expert_id]
            if task == 'TASK1':
                likelihood = data[0]
                likelihood = torch.softmax(likelihood, dim=-1)
                likelihood = torch.max(likelihood, dim=-1)[0]
            else:
                likelihood = data[0]
                likelihood = torch.softmax(likelihood, dim=-1)
            weight.append(likelihood[_index])
            preds.append(data[1][_index])
        weight = torch.stack(weight)
        preds = torch.stack(preds)
        pred_weight = preds * weight
        preds = pred_weight.sum(0) / weight.sum(0)
        preds = torch.greater(preds, 0.5).to(torch.long)
        all_preds.append(preds)
    all_preds = torch.stack(all_preds).to('cpu').numpy()
    labels = all_expert[0][-1].numpy()
    print(get_metric(labels, all_preds))


def main():
    for task in dataset_names:
        for dataset in dataset_names[task]:

            #用于处理新闻数据和专家知识。首先定义了两种专家：文本增强和图增强，每种都有不同的关注点。接着，它读取这些专家的模型，并将其存储在
            #逐条新闻生成提示信息，包括该新闻和相关专家的描述。接下来，它向一个函数
            #`get_reply`发送提示，以获取哪些专家知识是理解该新闻所需的，并将结果存储在输出列表中。最后，结果被保存在之前创建的文件中。
            run(task, dataset)

            #主要用于评估某个任务和数据集的模型表现。它的主要步骤如下：1. 定义文本和图形增强的类型。2. 从文件中加载每种增强方法下的模型预测结果。
            #3. 读取专家信息，这些信息包含每个专家的预测。4. 对于每个专家，提取他们的预测和权重。5. 计算所有专家的加权平均预测，并根据阈值将结果转化为类别。6. 最后，计算并打印实际标签与预测结果之间的指标。
            evaluate(task, dataset)

#多问一句，那些专家是你需要的
if __name__ == '__main__':
    main()
