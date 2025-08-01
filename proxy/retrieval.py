import json
import os

import nltk


dataset_names = {
    'TASK1': ['Pheme'],
    # 'TASK1': ['LLM-mis', 'Pheme'],
    # 'TASK2': ['MFC', 'SemEval-23F'],
    # 'TASK3': ['Generated', 'SemEval-20', 'SemEval-23P']
}


def main():
    for task in dataset_names:
        for dataset in dataset_names[task]:
            data = json.load(open(f'../data/datasets/{task}/{dataset}.json', encoding='utf-8'))
            wiki = json.load(open(f'../data/proxy/entity_from_wiki/{task}_{dataset}.json', encoding='utf-8'))
            save_dir = f'../data/proxy/retrieval/{task}_{dataset}.json'
            out = []
            for item, each_wiki in zip(data, wiki):
                out_text = item[0]
                for entity, exp in each_wiki:
                    if exp is None:
                        continue
                    exp = ' '.join(nltk.sent_tokenize(exp)[:3])
                    out_text=str(out_text)
                    entity_index = out_text.lower().find(entity.lower())
                    if entity_index != -1:
                        entity_index = entity_index + len(entity)
                        out_text = out_text[:entity_index] + ' (' + exp + ')' + out_text[entity_index:]
                out.append(out_text)
            if not os.path.exists('../data/proxy/retrieval'):
                os.mkdir('../data/proxy/retrieval')
            with open(save_dir, 'w', encoding='utf-8') as f:
                json.dump(out, f, ensure_ascii=False)

            #为每个数据集中的文本添加从维基百科获取的实体解释。具体步骤如下：
#2. 读取每个数据集和对应的维基百科数据。3. 创建一个保存路径，用于存放处理后的结果。
#4. 维基百科条目：- 获取原始文本。- 检查每个实体及其解释。-取前三句话。- 在文本中查找实体，若找到，则将说明添加到文本中。5. 将处理后的文本输出到指定文件中。
#最终，所有的结果被保存为 JSON 格式。

if __name__ == '__main__':
    main()
