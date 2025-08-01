import os
import json

import torch
from tqdm import tqdm
from utils import intent_detection, framing_detection, propaganda_detection, find_entity, intent_detection, \
    text_intent_detection, net_intent_detection, sentiment_detection, rep_generation

dataset_names = {
     'TASK1': ['weibo21']
}


def main():
    for task in dataset_names:
        for dataset in dataset_names[task]:
            data = json.load(open(f'../data/datasets/{task}/{dataset}.json',encoding='utf-8'))
            #  run the sentiment analysis proxy task
            if not os.path.exists('../data'):
                os.mkdir('../data')
            if not os.path.exists('../data/proxy'):
                os.mkdir('../data/proxy')
            if not os.path.exists('../data/proxy/intent'):
                os.mkdir('../data/proxy/intent')
            save_dir = f'../data/proxy/intent/{task}_{dataset}.pt'
            if os.path.exists(save_dir):
                out = torch.load(save_dir)
            else:
                out = []
            #中途要是断了，可以继续
            for item in tqdm(data[len(out):], leave=False, desc=f'{task}_{dataset}intent'):
                news_article = item[0]
                net_data = json.load(open(f'../data/networks/{task}_{dataset}/{len(out)}.json', encoding='utf-8'))
                #  truncate news article if needed
                text_intent = text_intent_detection(news_article)
                # text_intent = "作者对美团自动开通买单功能表示强烈不满，质疑该公司未征求用户同意便做出决定，认为这种做法损害了消费者权益。通过幽默的方式，作者希望引发公众关注，并促使公司对此进行反思和改进"

                net_intent = net_intent_detection(net_data)
                intent = intent_detection(text_intent,net_intent)
                intent_rep=rep_generation(intent)
                out.append(intent_rep)
                torch.save(out,save_dir)

            if not os.path.exists('../data/proxy/text_rep'):
                os.mkdir('../data/proxy/text_rep')
            save_dir = f'../data/proxy/text_rep/{task}_{dataset}.pt'
            if os.path.exists(save_dir):
                out = torch.load(save_dir)
            else:
                out = []
            #中途要是断了，可以继续
            for item in tqdm(data[len(out):], leave=False, desc=f'{task}_{dataset}text_rep'):
                news_article = item[0]
                #  truncate news article if needed
                text_rep = rep_generation(news_article)
                out.append(text_rep)
                torch.save(out,save_dir)

            if not os.path.exists('../data/proxy/comment_rep'):
                os.mkdir('../data/proxy/comment_rep')
            save_dir = f'../data/proxy/comment_rep/{task}_{dataset}.pt'
            if os.path.exists(save_dir):
                out = torch.load(save_dir)
            else:
                out = []
            #中途要是断了，可以继续
            for item in tqdm(data[len(out):], leave=False, desc=f'{task}_{dataset}comment_rep'):
                net_data = json.load(open(f'../data/networks/{task}_{dataset}/{len(out)}.json', encoding='utf-8'))
                comment = net_data['res']
                #  truncate news article if needed
                text_rep = rep_generation(comment)
                out.append(text_rep)
                torch.save(out,save_dir)
            # for item in tqdm(data[len(out):], leave=False, desc=f'{task}_{dataset}_sentiment'):
            #     news_article = item[0]
            #         #  truncate news article if needed
            #     sentiment = sentiment_detection(news_article)
            #     out.append(sentiment)
            #     with open(save_dir, 'w', encoding='utf-8') as f:
            #         json.dump(out, f, ensure_ascii=False)
            #
            #         #  run the framing detection proxy task
            # if not os.path.exists('../data/proxy/framing'):
            #     os.mkdir('../data/proxy/framing')
            # save_dir = f'../data/proxy/framing/{task}_{dataset}.json'
            # if os.path.exists(save_dir):
            #     out = json.load(open(save_dir))
            # else:
            #     out = []
            # for item in tqdm(data[len(out):], leave=False, desc=f'{task}_{dataset}_framing'):
            #     news_article = item[0]
            #         #  truncate news article if needed
            #     framing = framing_detection(news_article)
            #     out.append(framing)
            #     with open(save_dir, 'w', encoding='utf-8') as f:
            #         json.dump(out, f, ensure_ascii=False)
            #
            #     #  run the propaganda tactics detection proxy task
            # if not os.path.exists('../data/proxy/propaganda'):
            #     os.mkdir('../data/proxy/propaganda')
            # save_dir = f'../data/proxy/propaganda/{task}_{dataset}.json'
            # if os.path.exists(save_dir):
            #     with open(save_dir, 'w', encoding='utf-8') as f:
            #         json.dump(out, f, ensure_ascii=False)
            # else:
            #     out = []
            # for item in tqdm(data[len(out):], leave=False, desc=f'{task}_{dataset}_propaganda'):
            #     news_article = item[0]
            #         #  truncate news article if needed
            #     propaganda = propaganda_detection(news_article)
            #     out.append(propaganda)
            #     with open(save_dir, 'w', encoding='utf-8') as f:
            #         json.dump(out, f, ensure_ascii=False)
            #
            #     #  find the helpful entities for understanding the news
            if not os.path.exists('../data/proxy/entity'):
                os.mkdir('../data/proxy/entity')
            save_dir = f'../data/proxy/entity/{task}_{dataset}.json'
            if os.path.exists(save_dir):
                out=json.load(open(save_dir,encoding='utf-8'))
            else:
                out = []
            for item in data[len(out):]:
                news_article = item[0]
                    #  truncate news article if needed
                entities = find_entity(news_article)
                out.append(entities)
                with open(save_dir, 'w', encoding='utf-8') as f:
                    json.dump(out, f, ensure_ascii=False)
            #
            #     #这个代码的主要功能是处理一组数据集，并对每个数据集执行多个任务，包括情感分析、框架检测、宣传策略检测和实体识别。
if __name__ == '__main__':
    main()
