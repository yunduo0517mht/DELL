import json
import os
import torch
from utils import get_reply
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import re
from ensemble_utils import likelihood_weight


dataset_names = {
    'TASK1': ['Pheme']
    # 'TASK1': ['LLM-mis', 'Pheme'],
    # 'TASK2': ['MFC', 'SemEval-23F'],
    # 'TASK3': ['Generated', 'SemEval-20', 'SemEval-23P']
}


# def get_metric(y_true, y_pred):
#     return f1_score(y_true, y_pred, average='macro') * 100, \
#         f1_score(y_true, y_pred, average='micro') * 100
def get_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro') * 100, \
                f1_score(y_true, y_pred, average='micro') * 100, \
                    accuracy_score(y_true, y_pred) * 100

#这个代码的功能是处理一些新闻数据，并通过不同专家的分析来预测新闻的标签。代码的主要步骤包括：

#2. 从特定路径加载任务和数据集的文件。
#3. 定义不同类型的文本和图形增强方法，以及每个专家的描述。
#4. 依次加载每个专家的预测结果。
#6. 对于测试索引中剩下的每个新闻，提取新闻内容并生成提示信息，包含专家的分析。
#7. 调用 `get_reply` 函数得到最终的标签预测，并保存结果。
def run(task, dataset):
    if not os.path.exists('results'):
        os.mkdir('results')
    save_dir = f'results/vanilla_{task}_{dataset}.json'
    with open(f'../data/datasets/{task}/{dataset}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(f'D:\桌面\work\code\DELL-main(主题一版） - 副本\data\split/{task}_{dataset}/test.json' , 'r', encoding='utf-8') as f:
        test_indices = json.load(f)

    # experts and description of each expert
    text_augmentation = ["intent"]
    graph_augmentation = [None]

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
        expert = torch.load(f'../expert/results/{task}_{dataset}_{text}_{graph}_test.pt')
        all_experts.append(expert)
    if os.path.exists(save_dir):
        out = json.load(open(save_dir))
    else:
        out = []
    for _, index in enumerate(tqdm(test_indices[len(out):])):
        news = data[0][0]
        # truncate news article if needed
        prompt = 'News:\n{}\n\n'.format(news)
        prompt += 'Some experts give predictions about the news.\n'
        for _index in range(len(all_experts)):
            expert = all_experts[_index]
            expert_ans = expert[1][_]
            expert_ans = expert_ans.data.numpy()

            expert_info = 'Expert {}:\n'.format(_index + 1)
            expert_info += expert_descriptions[_index]

            expert_info += f'The expert predicts the label of this news is {expert_ans}.\n'
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


def evaluate(task, dataset):
    data = torch.load('../expert/results/{}_{}_None_None_test.pt'.format(task, dataset))
    # data = json.load(open(f'../data/datasets/{task}/{dataset}.json', encoding='UTF-8'))
    label = data[-1].numpy().tolist()
    results = json.load(open(f'results/vanilla_{task}_{dataset}.json'))
    # default_preds = likelihood_weight(task, dataset).tolist()
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
                # pred = default_preds[index]
                print("报错啦")
            preds.append(pred)
        else:
            if '1' in item:
                preds.append(1)
            elif '0' in item:
                preds.append(0)
            else:
                # preds.append(default_preds[index])
                print("报错啦")
    print(get_metric(label, preds))


def main():
    for task in dataset_names:
        for dataset in dataset_names[task]:
            run(task, dataset)

            evaluate(task, dataset)


if __name__ == '__main__':
    main()
