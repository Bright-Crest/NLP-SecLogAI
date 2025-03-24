#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练入口文件
用于训练NLP模型（BERT/GPT）进行日志分析和异常检测
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models.log_classifier import LogClassifier
from data.log_dataset import LogDataset
from utils.preprocessing import preprocess_logs


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练日志分析模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='从检查点恢复训练')
    return parser.parse_args()


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_model(config, checkpoint_path=None):
    """训练模型的主函数"""
    # 设置随机种子
    pl.seed_everything(config['training']['seed'])
    
    # 数据预处理
    train_data, val_data = preprocess_logs(
        log_path=config['data']['train_path'],
        val_split=config['data']['val_split']
    )
    
    # 创建数据集
    train_dataset = LogDataset(
        data=train_data,
        tokenizer_name=config['model']['pretrained_model_name'],
        max_length=config['model']['max_length']
    )
    
    val_dataset = LogDataset(
        data=val_data,
        tokenizer_name=config['model']['pretrained_model_name'],
        max_length=config['model']['max_length']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=RandomSampler(train_dataset),
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    # 创建模型
    model = LogClassifier(
        pretrained_model_name=config['model']['pretrained_model_name'],
        num_labels=config['model']['num_labels'],
        learning_rate=config['training']['learning_rate']
    )
    
    # 创建检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['checkpoint_dir'],
        filename='{epoch}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    # 早停回调
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config['training']['early_stop_patience'],
        mode='min'
    )
    
    # TensorBoard日志
    logger = TensorBoardLogger(
        save_dir=config['training']['log_dir'],
        name=config['model']['name']
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        gradient_clip_val=config['training'].get('gradient_clip_val', 0.0)
    )
    
    # 训练模型
    if checkpoint_path:
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, train_loader, val_loader)
    
    # 测试模型
    trainer.test(model, val_loader)
    
    # 保存最终模型
    model_save_path = os.path.join(config['training']['model_dir'], f"{config['model']['name']}_final.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至: {model_save_path}")


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    train_model(config, args.checkpoint) 