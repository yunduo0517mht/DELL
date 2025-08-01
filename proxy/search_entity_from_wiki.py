#  this file processes the output of find entity, and retrival related knowledge from wikipedia

import json
import wikipedia
import os
from tqdm import tqdm


dataset_names = {
    'TASK1': ['weibo21']
    # 'TASK1': ['LLM-mis', 'Pheme'],
    # 'TASK2': ['MFC', 'SemEval-23F'],
    # 'TASK3': ['Generated', 'SemEval-20', 'SemEval-23P']
}
# os.environ['http_proxy'] = 'http://127.0.0.1:15236'
# os.environ['https_proxy'] = 'http://127.0.0.1:15236'  # set the http proxy information
#

def main():
    for task in dataset_names:
        for dataset in dataset_names[task]:
            data = json.load(open(f'../data/proxy/entity/{task}_{dataset}.json',encoding='utf-8'))
            if not os.path.exists('../data/proxy/entity_from_wiki'):
                os.mkdir('../data/proxy/entity_from_wiki')
            save_dir = f'../data/proxy/entity_from_wiki/{task}_{dataset}.json'
            if os.path.exists(save_dir):
                with open(save_dir, 'r', encoding='utf-8') as f:
                    out = f.read()
            else:
                out = []
            for item in tqdm(data[len(out):]):
                item = item.split(']')[0]
                item = item[1:]
                item = item.split(',')
                item = [_.strip()[1:-1] for _ in item]
                item_res = []
                for _ in item:
                    try:
                        #搜索条目
                        wikipedia.set_lang("zh")
                        wiki_item = wikipedia.search(_)[0]
                    except Exception as e:
                        item_res.append([_,None])
                        # item_res.append(None)
                        continue
                    try:
                        content = wikipedia.summary(wiki_item, auto_suggest=False)
                    except wikipedia.DisambiguationError as e:
                        try:
                            s = e.options[0]
                            content = wikipedia.summary(s, auto_suggest=False)
                        except Exception:
                            item_res.append(None)
                            continue
                    except Exception:
                        item_res.append(None)
                        continue
                    item_res.append([_, content])
                out.append(item_res)
                with open(save_dir, 'w', encoding='utf-8') as f:
                    json.dump(out, f, ensure_ascii=False)  # ensure_ascii=False 以确保中文字符正常保存
                # json.dump(out, open(save_dir, 'w'))

#6. 尝试在维基百科上搜索每个项目，并获取摘要内容。如果出现错误，则记录为 None。
if __name__ == '__main__':
    main()
