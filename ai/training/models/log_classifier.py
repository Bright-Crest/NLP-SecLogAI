#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志分类模型
基于BERT的预训练模型，用于日志事件分类（如登录失败、攻击尝试等）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class LogClassifier(pl.LightningModule):
    """基于BERT的日志分类器，用PyTorch Lightning实现"""
    
    def __init__(self, pretrained_model_name='bert-base-uncased', num_labels=5, learning_rate=2e-5):
        """
        初始化模型
        
        Args:
            pretrained_model_name: 预训练模型名称，默认为bert-base-uncased
            num_labels: 分类标签数量
            learning_rate: 学习率
        """
        super(LogClassifier, self).__init__()
        self.save_hyperparameters()
        
        # 加载预训练BERT模型
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        
        # 分类器
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # 保存超参数
        self.learning_rate = learning_rate
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """前向传播"""
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取[CLS]令牌的表示（第一个位置）
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # 分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch.get('token_type_ids')
        labels = batch['labels']
        
        # 前向传播
        logits = self(input_ids, attention_mask, token_type_ids)
        
        # 计算损失
        loss = F.cross_entropy(logits, labels)
        
        # 记录训练损失
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch.get('token_type_ids')
        labels = batch['labels']
        
        # 前向传播
        logits = self(input_ids, attention_mask, token_type_ids)
        
        # 计算损失
        loss = F.cross_entropy(logits, labels)
        
        # 计算预测结果
        preds = torch.argmax(logits, dim=1)
        
        # 记录验证损失和指标
        self.log('val_loss', loss, prog_bar=True)
        
        return {'val_loss': loss, 'preds': preds, 'labels': labels}
    
    def validation_epoch_end(self, outputs):
        """验证周期结束"""
        # 收集所有批次的预测和标签
        preds = torch.cat([x['preds'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        
        # 转换为CPU NumPy数组
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # 计算指标
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        
        # 记录指标
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_f1', f1)
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        """测试周期结束"""
        return self.validation_epoch_end(outputs)
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 分层学习率
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # Cosine学习率调度
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            },
        }
    
    @staticmethod
    def load_from_checkpoint(checkpoint_path, map_location=None):
        """从检查点加载模型"""
        return super(LogClassifier, LogClassifier).load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location
        ) 