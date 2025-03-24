#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型加载器
负责加载和初始化训练好的模型，用于推理服务
"""

import os
import torch
from transformers import AutoTokenizer, AutoModel

# 尝试导入模型类
try:
    from training.models.log_classifier import LogClassifier
except ImportError:
    # 如果无法直接导入，则定义兼容的类
    import sys
    import pytorch_lightning as pl
    
    class LogClassifier(pl.LightningModule):
        """兼容的日志分类器类"""
        def __init__(self, pretrained_model_name='bert-base-uncased', num_labels=5, learning_rate=2e-5):
            super(LogClassifier, self).__init__()
            self.save_hyperparameters()
            self.bert = AutoModel.from_pretrained(pretrained_model_name)
            self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        
        def forward(self, input_ids, attention_mask, token_type_ids=None):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            return self.classifier(outputs.last_hidden_state[:, 0])


class ModelInferenceWrapper:
    """模型推理包装器，提供统一的预测接口"""
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化推理包装器
        
        Args:
            model: PyTorch模型
            tokenizer: 分词器
            device: 推理设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict(self, text):
        """
        对输入文本进行预测
        
        Args:
            text: 输入文本
            
        Returns:
            预测结果字典
        """
        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # 将编码移动到设备上
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        return self._process_outputs(outputs, text)
    
    def _process_outputs(self, outputs, text):
        """
        处理模型输出
        子类应重载此方法以实现特定处理逻辑
        """
        raise NotImplementedError("子类应实现此方法")


class ClassifierInferenceWrapper(ModelInferenceWrapper):
    """日志分类器推理包装器"""
    
    def _process_outputs(self, outputs, text):
        """处理分类器输出"""
        logits = outputs
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        prediction = torch.argmax(probs).item()
        
        # 标签映射
        label_map = {
            0: 'logon',
            1: 'logoff',
            2: 'connection_blocked',
            3: 'file_access',
            4: 'system_error'
        }
        
        return {
            'label': prediction,
            'label_name': label_map.get(prediction, 'unknown'),
            'confidence': probs[prediction].item(),
            'probabilities': {label_map.get(i, f'label_{i}'): prob.item() for i, prob in enumerate(probs)}
        }


class AnomalyInferenceWrapper(ModelInferenceWrapper):
    """异常检测推理包装器"""
    
    def __init__(self, model, tokenizer, threshold=0.7, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化异常检测包装器
        
        Args:
            model: PyTorch模型
            tokenizer: 分词器
            threshold: 异常阈值
            device: 推理设备
        """
        super(AnomalyInferenceWrapper, self).__init__(model, tokenizer, device)
        self.threshold = threshold
    
    def _process_outputs(self, outputs, text):
        """处理异常检测输出"""
        # 对于异常检测，我们假设模型输出一个异常分数
        score = outputs.item() if isinstance(outputs, torch.Tensor) and outputs.numel() == 1 else outputs[0].item()
        
        return {
            'is_anomaly': score > self.threshold,
            'score': score,
            'text': text
        }


def load_classifier_model(model_path, model_type='bert-base-uncased', num_labels=5):
    """
    加载日志分类模型
    
    Args:
        model_path: 模型路径
        model_type: 预训练模型类型
        num_labels: 标签数量
        
    Returns:
        分类器推理包装器
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    try:
        # 加载模型
        model = LogClassifier.load_from_checkpoint(model_path)
    except Exception as e:
        # 如果无法直接加载检查点，则尝试加载状态字典
        model = LogClassifier(pretrained_model_name=model_type, num_labels=num_labels)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    
    # 返回推理包装器
    return ClassifierInferenceWrapper(model, tokenizer)


def load_anomaly_model(model_path, model_type='bert-base-uncased', threshold=0.7):
    """
    加载异常检测模型
    
    Args:
        model_path: 模型路径
        model_type: 预训练模型类型
        threshold: 异常阈值
        
    Returns:
        异常检测推理包装器
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        # 创建一个简单的异常检测模型用于示例
        print(f"警告: 模型文件不存在 ({model_path})，创建一个简单的示例模型")
        
        class SimpleAnomalyModel(torch.nn.Module):
            """简单的异常检测模型（用于示例）"""
            def __init__(self):
                super(SimpleAnomalyModel, self).__init__()
                self.bert = AutoModel.from_pretrained(model_type)
                self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)
                self.sigmoid = torch.nn.Sigmoid()
            
            def forward(self, input_ids, attention_mask, token_type_ids=None):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                cls_output = outputs.last_hidden_state[:, 0]
                return self.sigmoid(self.classifier(cls_output))
        
        model = SimpleAnomalyModel()
    else:
        # 加载已有模型
        model = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    
    # 返回推理包装器
    return AnomalyInferenceWrapper(model, tokenizer, threshold=threshold) 