# we provide three non LLMs-based ensemble methods, namely, majority vote, likelihood weighted vote,
# and train on validation set.
import random
import torch
import torch.nn as nn
from einops import repeat
from sklearn.metrics import f1_score


def get_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro') * 100

#这个函数用于计算多标签分类的交叉熵损失。它的步骤如下：1. **调整预测值**：首先将预测值 `y_pred` 根据真实标签 `y_true` 进行一些调整，以帮助计算损失2. **创建负预测和正预测**：通过从预测值中减去一个大的值，分别得到负样本和正样本的预测结果。3. **添加零值**：在负和正预测数组的最后添加一个零值。
#4. **计算损失**：使用 `torch.logsumexp` 计算负样本和正样本的损失。5. **返回平均损失**：最后，返回所有样本的平均损失值。简而言之，这个函数帮助我们衡量多标签分类任务中预测与真实标签之间的差距。
def multilabel_categorical_cross_entropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    loss = neg_loss + pos_loss
    return loss.mean(0)


class MyModel(nn.Module):
    def __init__(self, k, task):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(k))
        if task == 'TASK1':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = multilabel_categorical_cross_entropy

    def forward(self, x, y):
        batch_size = len(x)
        weight = torch.softmax(self.weight, dim=0)
        weight = repeat(weight, 'n -> b n', b=batch_size)
        x = torch.einsum('bi, bij->bj', weight, x)
        pred = x

        loss = self.loss_fn(pred, y)
        return pred, loss

#这个函数叫做majority_vote，它用于进行投票预测。首先，它定义了文本和图像增强的类型。然后它创建了一个空的列表来存储预测结果和所有数据。接下来，它通过循环读取每种文本和图像增强类型的数据。读取的数据被存入all_data列表，并提取出预测结果存入preds列表。之后，它将所有预测结果堆叠在一起并计算它们的平均值。最后，它将平均值与0.5进行比较，得到最后的预测结果，并将其转换为numpy数组返回。
def majority_vote(task, dataset):
    text_augmentation = ['None', 'emotion', 'framing', 'propaganda', 'retrieval', 'None', 'None']
    graph_augmentation = ['None', 'None', 'None', 'None', 'None', 'stance', 'relation']
    preds = []
    all_data = []
    for text, graph in zip(text_augmentation, graph_augmentation):
        data = torch.load(f'../expert/results/{task}_{dataset}_{text}_{graph}_test.pt')
        all_data.append(data)
        preds.append(data[1])
    preds = torch.stack(preds).to(torch.float).mean(0)
    preds = torch.greater(preds, 0.5).to(torch.long)
    preds = preds.numpy()
    return preds

#这个函数的作用是计算任务和数据集相关的预测结果及其权重。
# 首先，它定义了文本和图形增强的方法。
# 然后，它会加载不同增强方法下的数据，并计算每种情况下的“似然性”得分。
# 接着，似然性得分通过softmax函数进行标准化，最终得到每个增强情况的预测结果和权重。
# 最后，函数会结合这些预测和权重，计算出最终的预测结果，并将其转换为长整形数组返回。
def likelihood_weight(task, dataset):
    text_augmentation = ['None', 'emotion', 'framing', 'propaganda', 'retrieval', 'None', 'None']
    graph_augmentation = ['None', 'None', 'None', 'None', 'None', 'stance', 'relation']
    preds = []
    weights = []
    all_data = []
    for text, graph in zip(text_augmentation, graph_augmentation):
        data = torch.load(f'../expert/results/{task}_{dataset}_{text}_{graph}_test.pt')
        all_data.append(data)
        if task == 'TASK1':
            likelihood = data[0]
            likelihood = torch.softmax(likelihood, dim=-1)
            likelihood = torch.max(likelihood, dim=-1)[0]
        else:
            likelihood = data[0]
            likelihood = torch.softmax(likelihood, dim=-1)
        preds.append(data[1])
        weights.append(likelihood)
    preds = torch.stack(preds)
    weights = torch.stack(weights)
    pred_weight = preds * weights
    preds = pred_weight.sum(0) / weights.sum(0)
    preds = torch.greater(preds, 0.5).to(torch.long)
    preds = preds.numpy()
    return preds


def train(model, train_x, train_y, val_x, val_y, task):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_acc = 0
    best_state = model.state_dict()
    for i in range(5000):
        model.train()
        optimizer.zero_grad()
        _, loss = model(train_x, train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        out, _ = model(val_x, val_y)
        if task == 'TASK1':
            preds = out.argmax(-1).to('cpu').numpy()
        else:
            preds = (out > 0).to(torch.long).to('cpu').numpy()
        label = val_y.to('cpu').numpy()
        acc = get_metric(label, preds)
        if acc > best_acc:
            best_acc = acc
            for key, value in model.state_dict().items():
                best_state[key] = value.clone()
    model.load_state_dict(best_state)

#该函数的主要功能是处理验证集并训练模型。它执行以下步骤：1. 初始化设备为CPU，并定义文本和图形增强的方法。2. 通过循环加载验证和测试数据。3. 根据任务类型，从加载的数据中提取特征和标签。4. 将验证数据随机分成训练集和验证集，比例为80%和20%。
#5. 将训练集、验证集和测试集的数据转移到设备上，并转换为浮点型。6. 创建模型并进行训练。7. 在测试集上进行预测，根据任务类型处理输出格式并返回预测结果。
def train_on_validation(task, dataset):
    device = torch.device('cpu')
    text_augmentation = ['None', 'emotion', 'framing', 'propaganda', 'retrieval', 'None', 'None']
    graph_augmentation = ['None', 'None', 'None', 'None', 'None', 'stance', 'relation']
    all_data = []
    all_data_val = []
    val_x = []
    test_x = []
    for text, graph in zip(text_augmentation, graph_augmentation):
        data = torch.load(f'../expert/results/{task}_{dataset}_{text}_{graph}_val.pt')
        if task == 'TASK1':
            val_x.append(data[0])
        else:
            val_x.append(data[1])
        all_data_val.append(data)
        data = torch.load(f'../expert/results/{task}_{dataset}_{text}_{graph}_test.pt')
        if task == 'TASK1':
            test_x.append(data[0])
        else:
            test_x.append(data[1])
        all_data.append(data)
    val_x = torch.stack(val_x).transpose(0, 1)
    val_y = all_data_val[0][-1]

    index = [_ for _ in range(len(val_x))]
    random.shuffle(index)
    train_index = index[:int(len(index) * 0.8)]
    val_index = index[int(len(index) * 0.8):]

    train_x, train_y = val_x[train_index].to(device).to(torch.float), val_y[train_index].to(device)
    val_x, val_y = val_x[val_index].to(device).to(torch.float), val_y[val_index].to(device)

    test_x = torch.stack(test_x).transpose(0, 1).to(device).to(torch.float)
    label = all_data[0][-1].to(device)
    model = MyModel(val_x.shape[-1], task).to(device)
    train(model, train_x, train_y, val_x, val_y, task)
    out, _ = model(test_x, label)
    if task == 'TASK1':
        preds = out.argmax(-1).to('cpu').numpy()
    else:
        preds = (out > 0).to(torch.long).to('cpu').numpy()
    return preds

