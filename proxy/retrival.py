import json
import nltk


dataset_names = {
    'TASK1': ['LLM-mis', 'Pheme'],
    'TASK2': ['MFC', 'SemEval-23F'],
    'TASK3': ['Generated', 'SemEval-20', 'SemEval-23P']
}


def main():
    for task in dataset_names:
        for dataset in dataset_names[task]:
            data = json.load(open(f'../data/datasets/{task}/{dataset}.json'))
            wiki = json.load(open(f'../data/proxy/entity_from_wiki/{task}_{dataset}.json'))
            save_dir = f'../data/proxy/retrieval/{task}_{dataset}.json'
            out = []
            for item, each_wiki in zip(data, wiki):
                out_text = item[0]
                for entity, exp in each_wiki:
                    if exp is None:
                        continue
                    exp = ' '.join(nltk.sent_tokenize(exp)[:3])
                    entity_index = out_text.lower().find(entity.lower())
                    if entity_index != -1:
                        entity_index = entity_index + len(entity)
                        out_text = out_text[:entity_index] + ' (' + exp + ')' + out_text[entity_index:]
                out.append(out_text)
            json.dump(out, open(save_dir, 'w'))

#处理多个数据集中的数据。读取每个数据集与之相关的维基百科信息。然后，它将这些数据进行加工
# 将维基百科中的实体和它们的描述插入到原始文本中。最终，程序将加工后的文本保存回一个新的JSON文件
if __name__ == '__main__':
    main()
