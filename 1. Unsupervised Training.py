#!/usr/bin/env python
# coding: utf-8
# %%
## 1. Training Unsupervised SentenceTransformer

# %%

import faulthandler
faulthandler.enable()

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # jupyter 环境设置


# %%
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sentence_transformers import datasets


from datasets import Dataset


import warnings
warnings.filterwarnings('ignore')


# %%
# Custom libraries
from utils.unsupervised_utils import read_data
from utils.utils import read_config
from utils.evaluators import InformationRetrievalEvaluator
# %%
os.environ["TOKENIZERS_PARALLELISM"]="true" # tokens并行
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"]="false" # 打开性能警告


# %%
config = read_config() # 读取配置文件


# %%
DATA_PATH = "../raw_data/" # 数据路径


# %%
# 读取训练数据
topics, content, correlations, _ = read_data(data_path=DATA_PATH, config_obj=config, read_mode="train")

# topic 和 content 分别加上列名前缀
topics.rename(columns=lambda x: "topic_" + x, inplace=True)
content.rename(columns=lambda x: "content_" + x, inplace=True)

# groud truth
correlations["content_id"] = correlations["content_ids"].str.split(" ")
corr = correlations.explode("content_id").drop(columns=["content_ids"])

# groud truth 和 topic, content 的合并
corr = corr.merge(topics, how="left", on="topic_id")
corr = corr.merge(content, how="left", on="content_id")

# topic_input, content_input -> df  
corr["set"] = corr[["topic_model_input", "content_model_input"]].values.tolist()
train_df = pd.DataFrame(corr["set"])

dataset = Dataset.from_pandas(train_df)

train_examples = []
train_data = dataset["set"]
n_examples = dataset.num_rows # 训练样本数量

# 构造 train_examples
for i in range(n_examples):
    example = train_data[i]
    if example[0] == None:
        continue        
    train_examples.append(InputExample(texts=[str(example[0]), str(example[1])]))

# %%
# Setting-up the Evaluation
# 读取验证数据
test_topics, test_content, test_correlations, _ = read_data(data_path=DATA_PATH, config_obj=config, read_mode="test")

test_correlations["content_id"] = test_correlations["content_ids"].str.split(" ")
test_correlations = test_correlations[test_correlations.topic_id.isin(test_topics.id)].reset_index(drop=True)
test_correlations["content_id"] = test_correlations["content_id"].apply(set)
test_correlations = test_correlations[["topic_id", "content_id"]] # 保留groud truth的 topic_id 和 content_id

# validation gt 存成字典: {topic_id: content_id}
ir_relevant_docs = {
    row['topic_id']: row['content_id'] for i, row in tqdm(test_correlations.iterrows())
}


# %%
# 保留不重复的 topic_id
unq_test_topics = test_correlations.explode("topic_id")[["topic_id"]].reset_index(drop=True).drop_duplicates().reset_index(drop=True) 
# 唯一个topic_id，合并上对应的 model_input
unq_test_topics = unq_test_topics.merge(test_topics[["id", "model_input"]], how="left", left_on="topic_id", right_on="id").drop("id", 1)

# validation 训练文本 存成字典: {topic_id: model_input}
ir_queries = {
    row['topic_id']: row['model_input'] for i, row in tqdm(unq_test_topics.iterrows())
}


# %%
# 读取全量数据
all_topics, all_content, _, special_tokens = read_data(data_path=DATA_PATH, config_obj=config, read_mode="all")
# 保留不重复的 content_id
unq_contents = correlations.explode("content_id")[["content_id"]].reset_index(drop=True).drop_duplicates().reset_index(drop=True)
# 唯一个content_id，合并上对应的 model_input
unq_contents = unq_contents.merge(all_content[["id", "model_input"]], how="left", left_on="content_id", right_on="id").drop("id", 1)
# 全量content 文本 存成字典: {content_id: model_input}
ir_corpus = {
    row['content_id']: row['model_input'] for i, row in tqdm(unq_contents.iterrows())
}

# %%
evaluator = InformationRetrievalEvaluator(
    ir_queries,  # validation topic_id 文本
    ir_corpus,   # 全量 content 文本
    ir_relevant_docs, # validation gt
    show_progress_bar=True, # 显示进度条
    main_score_function="cos_sim", # 主要得分函数
    precision_recall_at_k=[5, 10, 25, 50, 100],  # 精确率@k
    name='K12-local-test-unsupervised' # 评估器名称
)

# %%
# Training
# datasets.NoDuplicatesDataLoader 可以过滤掉重复数据，确保每个句子对只被输入模型一次。
train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=config["unsupervised_model"]["batch_size"])


# %%
TARGET_MODEL = config["unsupervised_model"]["base_name"] # 预训练模型
OUT_MODEL = config["unsupervised_model"]["save_name"] # 输出模型


# %%
model = SentenceTransformer(TARGET_MODEL) # 加载预训练模型
model.max_seq_length = config["unsupervised_model"]["seq_len"] # 设置maxlen

word_embedding_model = model._first_module() # 获取词向量模型

# 添加 sep token 到 tokenizer 中，并重新调整 token 的数量
word_embedding_model.tokenizer.add_tokens(list(special_tokens), special_tokens=True)
word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))


# %%
train_loss = losses.MultipleNegativesRankingLoss(model=model) # 定义损失函数

#k% of train data
num_epochs = config["unsupervised_model"]["epochs"] # 训练轮数
warmup_steps = int(len(train_dataloader) * config["unsupervised_model"]["warmup_ratio"]) # 预热步数


# %%
model.fit(train_objectives=[(train_dataloader, train_loss)], # 训练数据和损失函数
#           scheduler="constantlr",
#           optimizer_class=Lion,
#           optimizer_params={'lr': 2e-5},
          evaluator=evaluator,  # 定义评估器
#           evaluation_steps=400,
          
          checkpoint_path=f"checkpoints/unsupervised/{OUT_MODEL.split('/')[-1]}", # 保存检查点的路径
          checkpoint_save_steps=len(train_dataloader), # 保存检查点的步数
    
          epochs=num_epochs, 
          warmup_steps=warmup_steps,
          output_path=OUT_MODEL,
          save_best_model=True,
          use_amp=True # 混合精度训练
          )

