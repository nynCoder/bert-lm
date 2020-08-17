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


### test-case



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




