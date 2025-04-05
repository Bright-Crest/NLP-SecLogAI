# 📁 基于 BERT-mini + Fixed & Sliding Windows + 无盒标学习的日志异常检测项目

---

## 一、🚀 项目总览

本项目旨在使用较为轻量级的 BERT-mini 模型，配合 fixed windows 和 sliding windows 方法构造日志行为序列，实现无需标注的日志异常检测系统，并保持可简易部署和提供强扩展性。

---

## 二、🔢 核心技术路线

| 组件 | 技术路线 |
|--------|------------|
| 模型基库 | `prajjwal1/bert-mini` (HuggingFace) |
| 日志分组 | Fixed Windows + Sliding Windows |
| 学习方式 | MLM (自相关) + Embedding 距离 / Score 方差分析 |
| 无需标注 | 全部正常日志进行自相关训练 |
| 异常量化 | ROC-AUC + t-SNE + 分布分析 |

---

## 三、🌐 数据处理

### 输入日志格式：
根据 Apache/Linux 原始日志格式，每条日志为一条记录

```text
Oct 10 12:00:00 sshd[1000]: Failed password for root from 192.168.1.10
```

### 预处理 (log_tokenizer.py):
- 移除时间、loglevel
- 把 IP 和 用户名等变量符号化

```text
→ "failed password for user from ip"
```

### 分组方式
#### ὓ9 Fixed Windows:
- 每 N=10 条日志为一组
```python
[log0, log1, ..., log9] → 拼接为 "log0 [SEP] log1 [SEP] ..."
```

#### ὓ9 Sliding Windows:
- 每步长 1 ，切割短帧
```python
[log0:log9], [log1:log10], ...
```

---

## 四、📚 模型结构

### 基础模型: BERT-mini
- 4 层 Transformer
- 11M 参数，适合 RTX 3050Ti

### 输出头: MLP Anomaly Score Head
```text
[CLS] → Linear(256→64) → ReLU → Linear(64→1) → sigmoid
```

### 学习方式:
| 任务 | 说明 |
|--------|---------|
| MLM | Masked Token 预添入训练 |
| Embedding 差值 | 构造平均向量，距离过大则异常 |
| Score 方差 | 用精度给出评分

---

## 五、🏋️ 训练流程

### Step 1 ：输入文本构造
```python
input_text = "log1 [SEP] log2 [SEP] ..."
tokens = tokenizer(input_text)
```

### Step 2 ：MLM 自相关训练
```python
Trainer(..., mlm=True, data_collator=MaskCollator)
```

### Step 3 ：异常得分评估
```python
score = model(input_ids)
sigmoid(score) → [0,1]
```

### Step 4 ：问题分类 (可选)
- 根据 score 给出分级 (例如 >0.85 归为 high-risk)

---

## 六、🎯 评估指标

| 指标 | 方法 |
|--------|--------|
| ROC-AUC | sklearn.roc_auc_score |
| t-SNE 分布 | sklearn.manifold.TSNE + [CLS] embedding |
| 异常分数分布 | 输出分数 hist |
| 價值差值排序 | 排序 score 查看 top-N 日志 |

---

## 七、📢 上线 API

### Flask API
```json
POST /score_log
{
  "log_seq": ["login failed from 1.1.1.1", "sudo reboot"]
}
→ { "score": 0.82, "is_anomaly": true }
```

### CLI
```bash
python ai_detect_cli.py --logs "log1" "log2" --window-type sliding
```

---

## 八、🔄 扩展描述

- 支持类 DeepLog 模型作为方向一
- 支持加入 AutoEncoder / KNN score fusion 评估
- 开放接入 GNN 主客体分析机制
- 提供 GPT-解释异常故障

---

## 结言

本项目采用简洁形式实现了无标注日志异常分析系统，采用形式化分组手段构造行为序列，保证体系体积汇创新。同时为后续加入解释性、行为分类和环境应对措施创造了基础。

