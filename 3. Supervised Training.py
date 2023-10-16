#!/usr/bin/env python
# coding: utf-8
# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # jupyter 环境设置


# %%
# =========================================================================================
# Libraries
# =========================================================================================
import gc
import time
import math
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


from sentence_transformers import SentenceTransformer, CrossEncoder, util
from sentence_transformers.readers import InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader

# Custom libraries
from utils.unsupervised_utils import read_data
from utils.utils import read_config
from utils.metrics import get_pos_score, get_f2_score

os.environ["TOKENIZERS_PARALLELISM"]="false" # tokens并行
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"]="true" # 关闭性能警告

# %%
config = read_config() # 读取配置文件
DATA_PATH = "../raw_data/"
GENERATED_DATA_PATH = "./generated_files/"


# %%
# 读取训练数据
train_df = pd.read_parquet(GENERATED_DATA_PATH + "unsupervised_train.parquet")
test_df = pd.read_parquet(GENERATED_DATA_PATH + "unsupervised_test.parquet")

correlation_df = pd.read_csv(DATA_PATH + "correlations.csv")

# %%
# 构建 Datasets
train_samples = [InputExample(texts=[row.model_input1, row.model_input2],
                              label=int(row.target)) for row in tqdm(train_df.itertuples())]

test_samples = [InputExample(texts=[row.model_input1, row.model_input2],
                              label=int(row.target)) for row in tqdm(test_df.itertuples())]


# %%
# 创建一个 CrossEncoder 模型
# CrossEncoder 模型是一个二分类模型，用于判断两个句子是否相关
model = CrossEncoder(config["supervised_model"]["base_name"],  # 预训练模型名称
                     num_labels=1, # 指定模型输出的类别数量。在这个例子中，我们只关心两个句子之间的关系得分，所以设置为 1。
                     max_length=config["supervised_model"]["seq_len"] # maxlen
                     )

num_epochs = config["supervised_model"]["epochs"] # epochs

# train dataloader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=config["supervised_model"]["batch_size"], num_workers=0, pin_memory=False)

# 创建 evaluator
evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_samples, name='K12-local-test', show_progress_bar=True)

# warmup steps
warmup_steps = math.ceil(len(train_dataloader) * config["supervised_model"]["warmup_ratio"]) 


# %%
# 训练模型
model.fit(train_dataloader=train_dataloader, # 训练数据
          show_progress_bar=True,
          evaluator=evaluator, # 评估器
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          save_best_model=True,
          output_path=config["supervised_model"]["save_name"],
          use_amp=True # 混合精度
          )


# %%
# 加载已保存的模型
model = CrossEncoder(config["supervised_model"]["save_name"], # 已保存的模型
                    num_labels=1, # 指定模型输出的类别数量。在这个例子中，我们只关心两个句子之间的关系得分，所以设置为 1。
                    max_length=config["supervised_model"]["seq_len"]
                    )


preds = model.predict(test_df[["model_input1", "model_input2"]].values, # 验证数据
                      show_progress_bar=True,
                      batch_size=96
                      )

test_df["pred_score"] = preds # 保存验证集的预测


# %%
# 查看不同下的阈值F2分数，在之后的inference中，我们可以参考这个阈值
for thr in np.arange(0., 0.3, 0.0025):
    # 根据当前 thr 筛选出预测值大于等于thr的数据，并按预测值降序排序。
    # 按照 topics_ids 对筛选后的数据进行分组，并将每个组内的 content_ids 进行拼接，然后重命名结果列为 "pred_content_ids"。
    preds_thr_df = test_df[test_df.pred_score >= thr].sort_values(by="pred_score",ascending=False)[["topics_ids","content_ids"]].\
                                    groupby("topics_ids")["content_ids"].apply(lambda x: " ".join(x)).rename("pred_content_ids").reset_index()
    # 将预测结果与 Groud Truth 数据（correlation_df）进行合并，以便后续计算 F2 分数。
    preds_thr_df = preds_thr_df.merge(correlation_df[correlation_df.topic_id.isin(test_df.topics_ids)],
                                      how="outer", right_on="topic_id", left_on="topics_ids")
    preds_thr_df.fillna("None", inplace=True) # 用None填充空值
    
    # 计算当前阈值下的 F2 分数
    f2score_for_threshold = get_f2_score(preds_thr_df['content_ids'], preds_thr_df['pred_content_ids'])

    print(f"Threshold: {thr} | Score: {f2score_for_threshold}")
