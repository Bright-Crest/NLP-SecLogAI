import torch
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, precision_recall_fscore_support
from transformers import BertModel, DistilBertModel, BertTokenizer
from app.models.tinylogbert_model import TinyLogBERT
from app.models.logbert_distil_model import LogBERTDistil
import os
import json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class LogDataset(Dataset):
    """简单的日志数据集，用于批量测试"""
    def __init__(self, logs, tokenizer, max_length=128):
        self.logs = logs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.logs)
    
    def __getitem__(self, idx):
        log = self.logs[idx]
        encoding = self.tokenizer(log["text"], 
                                 max_length=self.max_length,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")
        
        # 去掉batch维度
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(log["label"], dtype=torch.float),
            "text": log["text"]
        }

class ModelComparisonService:
    """
    模型比较服务，用于评估不同模型在日志异常检测上的性能表现
    支持对比TinyLogBERT和LogBERT-Distil
    """
    
    def __init__(self, results_dir="./comparison_results"):
        """
        初始化服务
        params:
            results_dir: 结果保存目录
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, model_name, model_path, model_type="tinylogbert"):
        """
        加载模型
        params:
            model_name: 模型名称（用于标识）
            model_path: 模型文件路径
            model_type: 模型类型 ("tinylogbert" 或 "logbert_distil")
        """
        try:
            if model_type.lower() == "tinylogbert":
                # 加载BERT-mini基础模型
                base_model = BertModel.from_pretrained('prajjwal1/bert-mini')
                model = TinyLogBERT(base_model)
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            elif model_type.lower() == "logbert_distil":
                # 加载DistilBERT模型
                base_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
                model = LogBERTDistil(base_model)
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 加载预训练权重
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"成功加载模型: {model_path}")
            else:
                print(f"警告: 模型文件{model_path}不存在，使用未训练的模型")
            
            # 移动模型到设备并设置为评估模式
            model.to(self.device)
            model.eval()
            
            # 保存模型和分词器
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            return True
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            return False
    
    def compare_models(self, test_data, batch_size=32, save_results=True):
        """
        比较多个模型的性能
        params:
            test_data: 测试数据 (列表，每个元素为字典 {"text": "日志文本", "label": 0/1})
            batch_size: 批大小
            save_results: 是否保存结果
        return:
            比较结果字典
        """
        if not self.models:
            raise ValueError("请先加载要比较的模型")
        
        results = {}
        
        # 为每个模型准备结果存储
        for model_name in self.models:
            results[model_name] = {
                "predictions": [],
                "labels": [],
                "inference_time": 0,
                "metrics": {}
            }
        
        # 创建通用数据加载器
        for model_name, model in self.models.items():
            tokenizer = self.tokenizers[model_name]
            dataset = LogDataset(test_data, tokenizer)
            dataloader = DataLoader(dataset, batch_size=batch_size)
            
            # 记录推理时间
            start_time = time.time()
            
            # 处理每个批次
            for batch in tqdm(dataloader, desc=f"评估 {model_name}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].numpy()
                
                # 预测
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask)
                    scores = torch.sigmoid(outputs).squeeze().cpu().numpy()
                
                # 保存预测结果
                results[model_name]["predictions"].extend(scores.tolist() if hasattr(scores, "tolist") else [scores])
                results[model_name]["labels"].extend(labels.tolist() if hasattr(labels, "tolist") else [labels])
            
            # 计算总推理时间
            inference_time = time.time() - start_time
            results[model_name]["inference_time"] = inference_time
            
            # 计算评估指标
            y_true = np.array(results[model_name]["labels"])
            y_score = np.array(results[model_name]["predictions"])
            
            # ROC AUC
            auc = roc_auc_score(y_true, y_score)
            
            # 使用0.5作为阈值的精确率、召回率和F1
            y_pred = (y_score >= 0.5).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            
            # 保存指标
            results[model_name]["metrics"] = {
                "auc": float(auc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "inference_time": float(inference_time),
                "inferences_per_second": len(test_data) / inference_time
            }
        
        # 汇总结果
        summary = {model_name: results[model_name]["metrics"] for model_name in self.models}
        
        # 保存结果
        if save_results:
            # 保存指标汇总
            summary_file = os.path.join(self.results_dir, "model_comparison_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
            print(f"模型对比汇总已保存到: {summary_file}")
            
            # 绘制ROC曲线对比
            self._plot_roc_comparison(results)
            
            # 绘制PR曲线对比
            self._plot_pr_comparison(results)
            
            # 绘制推理时间对比
            self._plot_inference_time(summary)
        
        return summary
    
    def _plot_roc_comparison(self, results):
        """绘制ROC曲线对比"""
        plt.figure(figsize=(10, 8))
        
        for model_name in results:
            y_true = np.array(results[model_name]["labels"])
            y_score = np.array(results[model_name]["predictions"])
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = results[model_name]["metrics"]["auc"]
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (False Positive Rate)')
        plt.ylabel('真阳性率 (True Positive Rate)')
        plt.title('接收者操作特征 (ROC) 曲线对比')
        plt.legend(loc="lower right")
        
        # 保存图像
        plt.savefig(os.path.join(self.results_dir, "roc_comparison.png"))
        plt.close()
    
    def _plot_pr_comparison(self, results):
        """绘制精确率-召回率曲线对比"""
        plt.figure(figsize=(10, 8))
        
        for model_name in results:
            y_true = np.array(results[model_name]["labels"])
            y_score = np.array(results[model_name]["predictions"])
            
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            
            plt.plot(recall, precision, label=f'{model_name}')
        
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title('精确率-召回率曲线对比')
        plt.legend(loc="best")
        
        # 保存图像
        plt.savefig(os.path.join(self.results_dir, "pr_comparison.png"))
        plt.close()
    
    def _plot_inference_time(self, summary):
        """绘制推理时间对比"""
        model_names = list(summary.keys())
        inference_times = [summary[model]["inference_time"] for model in model_names]
        inferences_per_second = [summary[model]["inferences_per_second"] for model in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # 推理总时间
        ax1.bar(model_names, inference_times)
        ax1.set_ylabel('推理时间 (秒)')
        ax1.set_title('总推理时间对比')
        
        # 每秒推理数量
        ax2.bar(model_names, inferences_per_second)
        ax2.set_ylabel('每秒推理次数')
        ax2.set_title('推理速度对比')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "inference_time_comparison.png"))
        plt.close()
    
    def compare_embeddings(self, logs, sample_size=200):
        """
        比较不同模型的嵌入向量
        params:
            logs: 日志列表
            sample_size: 用于可视化的样本数量
        """
        if len(logs) > sample_size:
            # 随机采样
            indices = np.random.choice(len(logs), sample_size, replace=False)
            logs_sample = [logs[i] for i in indices]
        else:
            logs_sample = logs
        
        embeddings = {}
        labels = []
        
        # 提取每个模型的嵌入向量
        for model_name, model in self.models.items():
            tokenizer = self.tokenizers[model_name]
            all_embeddings = []
            
            for log in tqdm(logs_sample, desc=f"提取{model_name}嵌入"):
                # 分词
                tokens = tokenizer(log["text"], padding='max_length', truncation=True,
                                max_length=128, return_tensors="pt")
                
                # 将输入移到正确的设备
                input_ids = tokens['input_ids'].to(self.device)
                attention_mask = tokens['attention_mask'].to(self.device)
                
                # 提取嵌入向量
                embedding = model.get_embedding(input_ids, attention_mask)
                all_embeddings.append(embedding.cpu().numpy())
                
                # 只需要为第一个模型保存标签
                if model_name == list(self.models.keys())[0]:
                    labels.append(log["label"])
            
            # 将嵌入向量堆叠为数组
            embeddings[model_name] = np.vstack([e.squeeze(0) for e in all_embeddings])
        
        # 可视化嵌入向量
        from sklearn.manifold import TSNE
        
        plt.figure(figsize=(15, 12))
        
        for i, model_name in enumerate(embeddings):
            # t-SNE降维
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings[model_name])
            
            # 创建子图
            plt.subplot(1, len(embeddings), i+1)
            
            # 绘制嵌入点
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=labels, cmap='coolwarm', alpha=0.7, s=40)
            
            plt.title(f'{model_name} 嵌入向量')
            plt.colorbar(scatter, label='标签')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "embedding_comparison.png"))
        plt.close()
        
        return embeddings

# 便于调用的函数
def compare_models(test_data, model_configs, batch_size=32):
    """
    比较多个模型的性能
    params:
        test_data: 测试数据列表
        model_configs: 模型配置列表，每个元素为字典，包含 name, path, type 字段
        batch_size: 批大小
    return:
        比较结果
    """
    comparison_service = ModelComparisonService()
    
    # 加载所有模型
    for config in model_configs:
        comparison_service.load_model(
            model_name=config["name"], 
            model_path=config["path"],
            model_type=config["type"]
        )
    
    # 比较模型性能
    return comparison_service.compare_models(test_data, batch_size=batch_size)

def compare_model_embeddings(logs, model_configs, sample_size=200):
    """
    比较不同模型的嵌入向量
    params:
        logs: 日志列表
        model_configs: 模型配置列表
        sample_size: 用于可视化的样本数量
    """
    comparison_service = ModelComparisonService()
    
    # 加载所有模型
    for config in model_configs:
        comparison_service.load_model(
            model_name=config["name"], 
            model_path=config["path"],
            model_type=config["type"]
        )
    
    # 比较嵌入向量
    return comparison_service.compare_embeddings(logs, sample_size) 