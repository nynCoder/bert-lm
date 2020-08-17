## BERT as Language Model

For a sentence <img src="https://www.zhihu.com/equation?tex=S%20=%20w_1,%20w_2,...,%20w_k" alt="S = w_1, w_2,..., w_k" eeimg="1"> , we have

<img src="https://www.zhihu.com/equation?tex=p(S)%20=%20\prod_{i=1}^{k}%20p(w_i%20|%20context)" alt="p(S) = \prod_{i=1}^{k} p(w_i | context)" eeimg="1"> 


In traditional language model, such as RNN,  <img src="https://www.zhihu.com/equation?tex=context%20=%20w_1,%20...,%20w_{i-1}" alt="context = w_1, ..., w_{i-1}" eeimg="1"> , 

<img src="https://www.zhihu.com/equation?tex=p(S)%20=%20\prod_{i=1}^{k}%20p(w_i%20|%20w_1,%20...,%20w_{i-1})" alt="p(S) = \prod_{i=1}^{k} p(w_i | w_1, ..., w_{i-1})" eeimg="1">


In bidirectional language model, it has larger context, <img src="https://www.zhihu.com/equation?tex=context+%3d+w_1%2c+...%2c+w_%7bi-1%7d%2cw_%7bi%2b1%7d%2c...%2cw_k" alt="context = w_1, ..., w_{i-1},w_{i+1},...,w_k" eeimg="1">.

In this implementation, we simply adopt the following approximation,

<img src="https://www.zhihu.com/equation?tex=p(S)+%5capprox+%5cprod_%7bi%3d1%7d%5e%7bk%7d+p(w_i+%7c+w_1%2c+...%2c+w_%7bi-1%7d%2cw_%7bi%2b1%7d%2c+...%2cw_k)" alt="p(S) \approx \prod_{i=1}^{k} p(w_i | w_1, ..., w_{i-1},w_{i+1}, ...,w_k)" eeimg="1">.


<!--
1. 近似相等
2. 句子越长，单个word预测的概率越大，ppl越大？传统的RNN也有这个问题
-->

<!-- n-gram
n-gram models construct tables of conditional probabilities for the next word,

Under Markov assumption, the context is the all the 
-->


### test-case 垂直领域：诗词



```bash
export BERT_BASE_DIR=model/chinese_L-12_H-768_A-12
export INPUT_FILE=data/lm/poetry2.tsv
python run_lm_predict.py \
  --input_file=$INPUT_FILE \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --output_dir=./tmp/lm_output/
```


$ cat /tmp/lm/output/test_result.json



### 上述代码是bert整个代码框架比较“大块”，当然有超级简单的bert调用方法：

```bash
'''BERT用作语言模型，计算句子分数，检验句子的合理性与否，
其实类似于基于bert-mlm的中文纠错，每个字符作为mask计算一个loss'''
from torch.multiprocessing import TimeoutError, Pool, set_start_method, Queue
import torch.multiprocessing as mp
import torch
import numpy as np
# from transformers import  DistilBertTokenizer,DistilBertForMaskedLM
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import json, math

try:
    set_start_method('spawn')
except RuntimeError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    ## 加载bert模型，这个路径文件夹下有bert_config.json配置文件和model.bin模型权重文件
    # bert-base-uncased是英文的
    model = BertForMaskedLM.from_pretrained('bert-base-chinese').to(device)
    model.eval()
    ## 加载bert的分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    return tokenizer, model


tokenizer, model = load_model()

'''
将loss作为句子困惑度ppl的分数:
不足：
1. 给每个word打分，都要跑一遍inference，计算量较大，且冗余。有优化的空间
2.该实现中采用的句子概率是近似概率，不够严谨
'''
def get_score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    # Predict all tokens
    predictions = model(tensor_input)  # model(masked_ids)
    #nn.CrossEntropyLoss(size_average=False)
    # 根据pytorch的官方文档，size_average默认情况下是True，对每个小批次的损失取平均值。 但是，如果字段size_average设置为False，则每个小批次的损失将被相加。如果参数reduce = False，则忽略
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(predictions.squeeze(), tensor_input.squeeze()).data#已经取平均值后的loss，作为句子的ppl分数返回
    return math.exp(loss)


print(get_score("杜甫是什么的诗词是有哪些"))
print(get_score("杜甫的诗词有哪些"))
```






