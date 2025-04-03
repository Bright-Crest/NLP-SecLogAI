#!/usr/bin/env python3
import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from transformers import BertModel, DistilBertModel, BertTokenizer
import logging
import time
from tqdm import tqdm
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models.tinylogbert_model import TinyLogBERT
from app.models.logbert_distil_model import LogBERTDistil
from app.services.log_tokenizer import LogTokenizer
from utils.log_converter import LogConverter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LogDataset(Dataset):
    """日志数据集"""
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
            "label": torch.tensor(log["label"], dtype=torch.float)
        }

def is_json_file(file_path):
    """
    检查文件是否为JSON格式
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 尝试读取第一行并解析为JSON
            line = f.readline().strip()
            data = json.loads(line)
            # 检查是否包含必要字段
            if isinstance(data, dict) and "text" in data and "label" in data:
                return True
    except (json.JSONDecodeError, UnicodeDecodeError, IOError):
        pass
    return False

def convert_log_file(file_path, anomaly_ratio=0.05):
    """
    将普通日志文件转换为JSON格式
    """
    logger.info(f"检测到普通日志文件，自动转换为JSON格式: {file_path}")
    
    # 创建转换器
    converter = LogConverter(anomaly_ratio=anomaly_ratio)
    
    # 转换文件
    output_path = converter.convert_file(file_path)
    
    return output_path

def load_data(data_file, sample_limit=None):
    """
    加载日志数据
    格式: 每行一条JSON，包含text和label字段
    如果是普通日志文件，自动转换为JSON格式
    """
    # 检查文件是否为JSON格式
    if not is_json_file(data_file):
        # 自动转换为JSON格式
        data_file = convert_log_file(data_file)
    
    logs = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            log = json.loads(line.strip())
            logs.append(log)
            if sample_limit and len(logs) >= sample_limit:
                break
    
    logger.info(f"加载了 {len(logs)} 条日志")
    return logs

def train_model(args):
    """训练模型"""
    # 设置随机种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(args.base_tokenizer)
    
    # 加载数据
    logger.info("加载数据...")
    logs = load_data(args.data_file, args.sample_limit)
    
    # 划分数据集
    train_size = int(len(logs) * 0.8)
    val_size = len(logs) - train_size
    train_logs, val_logs = random_split(logs, [train_size, val_size])
    
    # 创建数据集
    train_dataset = LogDataset(train_logs, tokenizer, args.max_length)
    val_dataset = LogDataset(val_logs, tokenizer, args.max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # 根据模型类型创建相应的模型
    logger.info(f"创建 {args.model_type} 模型...")
    
    if args.model_type.lower() == "tinylogbert":
        # 加载BERT-mini基础模型
        base_model = BertModel.from_pretrained('prajjwal1/bert-mini')
        model = TinyLogBERT(base_model)
        output_model_path = os.path.join(args.checkpoint_dir, "tiny_model.pt")
    elif args.model_type.lower() == "logbert_distil":
        # 加载DistilBERT模型
        base_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        model = LogBERTDistil(base_model)
        output_model_path = os.path.join(args.checkpoint_dir, "distil_model.pt")
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    model.to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练模型
    logger.info("开始训练...")
    best_val_loss = float("inf")
    early_stop_count = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                
                preds = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 计算指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 使用0.5作为阈值
        binary_preds = (all_preds >= 0.5).astype(int)
        accuracy = (binary_preds == all_labels).mean()
        
        # 打印训练信息
        elapsed_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, "
                   f"Accuracy: {accuracy:.4f}, "
                   f"Time: {elapsed_time:.1f}s")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_count = 0
            
            # 保存模型
            torch.save(model.state_dict(), output_model_path)
            logger.info(f"模型已保存到 {output_model_path}")
        else:
            early_stop_count += 1
            
        # 早停
        if early_stop_count >= args.patience:
            logger.info(f"早停: {args.patience} 个epoch没有改善")
            break
    
    logger.info("训练完成")
    return model

def main():
    parser = argparse.ArgumentParser(description="日志异常检测模型训练")
    
    # 数据参数
    parser.add_argument("--data-file", type=str, required=True, help="数据文件路径（支持普通日志文件或JSON格式）")
    parser.add_argument("--sample-limit", type=int, default=None, help="样本数量限制")
    parser.add_argument("--max-length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--anomaly-ratio", type=float, default=0.05, help="普通日志文件转换时的异常比例")
    
    # 模型参数
    parser.add_argument("--model-type", type=str, default="tinylogbert", 
                      choices=["tinylogbert", "logbert_distil"],
                      help="模型类型: tinylogbert 或 logbert_distil")
    parser.add_argument("--base-tokenizer", type=str, default="bert-base-uncased", help="分词器")
    
    # 训练参数
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--patience", type=int, default=3, help="早停耐心")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoint", help="检查点目录")
    parser.add_argument("--seed", type=int, help="随机种子")
    
    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.data_file):
        logger.error(f"错误: 数据文件不存在: {args.data_file}")
        return 1
    
    # 处理参数
    if args.seed is None:
        args.seed = random.randint(1, 100000000)
        print(f"使用随机种子: {args.seed}")

    # 训练模型
    train_model(args)

if __name__ == "__main__":
    main() 