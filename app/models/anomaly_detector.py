import os
import sys
import torch
import logging
import numpy as np
from transformers import BertTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.integrations import TensorBoardCallback
from transformers.trainer_callback import EarlyStoppingCallback
from torch.utils.data import Dataset
import json
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT_DIR, "ai_detect", "checkpoint")
OUTPUT_DIR = os.path.join(ROOT_DIR, "ai_detect", "output")

sys.path.append(ROOT_DIR)
from app.models.tinylogbert import create_tiny_log_bert
from app.models.log_window import LogWindow
from ai_detect.core.supervised_evaluator import evaluate_scores, visualize_results, find_best_threshold
from app.models.tinylogbert import AnomalyScoreEnsemble

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogDataset(Dataset):
    """日志数据集类，用于MLM训练"""
    
    def __init__(self, tokenizer, logs, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = self.tokenizer(logs, 
                                        truncation=True,
                                        max_length=max_length,
                                        padding='max_length',
                                        return_tensors='pt')
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item
    
    def __len__(self):
        return len(self.encodings.input_ids)


class ContrastiveLogDataset(Dataset):
    """支持对比学习的日志数据集类"""
    
    def __init__(self, tokenizer, logs, max_length=128, enable_augmentation=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_augmentation = enable_augmentation
        self.logs = logs
        
        # 对原始日志进行编码
        self.encodings = self.tokenizer(logs, 
                                      truncation=True,
                                      max_length=max_length,
                                      padding='max_length',
                                      return_tensors='pt')
        
        # 如果启用了数据增强，可以创建正样本对
        if enable_augmentation:
            self._create_positive_pairs()
    
    def _create_positive_pairs(self, ratio=0.5):
        """创建正样本对索引，选择一部分样本对进行配对"""
        num_samples = len(self.logs)
        num_pairs = int(num_samples * ratio)
        
        # 随机选择样本进行配对
        self.positive_indices = torch.tensor(
            random.sample(range(num_samples), min(num_pairs, num_samples)),
            dtype=torch.long
        )
        self.has_positive_pairs = len(self.positive_indices) > 0
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        
        # 如果启用了数据增强，并且当前样本被选为正样本对的一部分
        if self.enable_augmentation and self.has_positive_pairs:
            if idx in self.positive_indices:
                # 标记此样本需要进行数据增强
                item['augment'] = torch.tensor(1, dtype=torch.long)
            else:
                item['augment'] = torch.tensor(0, dtype=torch.long)
        
        return item
    
    def __len__(self):
        return len(self.encodings.input_ids)


class AnomalyDetector:
    """
    日志异常检测器，提供模型训练、评估和检测功能
    基于BERT-mini + Fixed/Sliding Windows + MLM 自监督学习
    增强版：支持MoCo风格的对比学习和数据增强
    无监督版：支持多种无监督异常检测方法
    融合版：支持多种异常检测方法的专家融合
    """
    
    DETECTION_METHODS = ['knn', 'cluster', 'lof', 'iforest', 'reconstruction', 'ensemble']
    
    def __init__(self, 
                 model_dir=MODEL_DIR,
                 output_dir=OUTPUT_DIR,
                 tokenizer_name="prajjwal1/bert-mini",
                 window_size=10,
                 detection_method='ensemble'):
        """
        初始化异常检测器
        
        参数:
            model_dir: 模型保存目录
            output_dir: 输出结果目录
            tokenizer_name: 使用的tokenizer名称
            window_size: 日志窗口大小
            detection_method: 异常检测方法，支持'knn', 'cluster', 'lof', 'iforest', 'reconstruction', 'ensemble'
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.tokenizer_name = tokenizer_name
        self.window_size = window_size
        
        # 确保检测方法有效
        if detection_method not in self.DETECTION_METHODS:
            logging.warning(f"不支持的检测方法: {detection_method}，使用默认方法: ensemble")
            detection_method = 'ensemble'
        self.detection_method = detection_method
        
        # 确保目录存在
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建TensorBoard日志目录
        self.tensorboard_dir = os.path.join(self.output_dir, "tensorboard")
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        # 初始化tokenizer和window处理器
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.window = LogWindow(tokenizer_name=tokenizer_name, window_size=window_size)
        
        # 初始化模型
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logging.info(f"异常检测器初始化完成，使用设备: {self.device}，检测方法: {detection_method}")
    
    def load_logs(self, log_file):
        """
        从文件加载日志
        
        参数:
            log_file: 日志文件路径
            
        返回:
            logs: 日志文本列表
        """
        logs = []
        encodings = ['utf-8', 'latin1', 'cp1252', 'gbk', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(log_file, 'r', encoding=encoding) as f:
                    logs = [line.strip() for line in f.readlines()]
                logging.info(f"成功使用 {encoding} 编码读取日志文件: {log_file}")
                break
            except UnicodeDecodeError:
                logging.warning(f"尝试使用 {encoding} 编码读取失败，尝试下一种编码")
            except Exception as e:
                logging.error(f"读取日志文件 {log_file} 失败: {str(e)}")
                raise
        
        if not logs:
            error_msg = f"无法使用支持的编码格式读取日志文件 {log_file}，尝试了以下编码: {', '.join(encodings)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
        return logs
    
    def prepare_log_windows(self, logs, window_type='fixed', stride=1):
        """
        准备日志窗口数据
        
        参数:
            logs: 日志文本列表
            window_type: 窗口类型 ('fixed' 或 'sliding')
            stride: 滑动窗口的步长
            
        返回:
            window_texts: 窗口文本列表
        """
        window_texts = []
        sep_token = self.tokenizer.sep_token
        
        if window_type == 'fixed':
            # 固定窗口
            for i in range(0, len(logs), self.window_size):
                end = min(i + self.window_size, len(logs))
                if end - i < self.window_size and i + self.window_size < len(logs):
                    continue  # 跳过不足一个窗口的最后部分
                
                window_logs = logs[i:end]
                window_text = f" {sep_token} ".join(window_logs)
                window_texts.append(window_text)
        else:
            # 滑动窗口
            if len(logs) < self.window_size:
                return window_texts
                
            for i in range(0, len(logs) - self.window_size + 1, stride):
                window_logs = logs[i:i + self.window_size]
                window_text = f" {sep_token} ".join(window_logs)
                window_texts.append(window_text)
        
        return window_texts
    
    # 自定义的数据校对器，支持数据增强和MoCo风格的对比学习
    class ContrastiveMLMCollator:
        def __init__(self, tokenizer, mlm_probability=0.15, enable_augmentation=True):
            self.tokenizer = tokenizer
            self.mlm_probability = mlm_probability
            self.enable_augmentation = enable_augmentation
            # 创建标准MLM数据校对器
            self.mlm_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=mlm_probability
            )
        
        def __call__(self, examples):
            # 处理MLM部分
            batch = self.mlm_collator([{k: v for k, v in example.items() if k not in ['augment']} 
                                     for example in examples])
            
            # 如果启用数据增强，添加augment标志
            if self.enable_augmentation and 'augment' in examples[0]:
                batch['augment_batch'] = torch.tensor([example.get('augment', 0) for example in examples], 
                                                    dtype=torch.bool)
            
            return batch
    
    def train(self, train_file, eval_file=None, num_epochs=10, batch_size=32, save_steps=500,
              early_stopping_patience=3, use_tensorboard=True, enable_contrastive=True, 
              enable_augmentation=True, collect_features=True, fusion_method='dynamic_weight'):
        """
        训练模型
        
        参数:
            train_file: 训练数据文件
            eval_file: 评估数据文件
            num_epochs: 训练轮次
            batch_size: 批次大小
            save_steps: 保存模型的步数间隔
            early_stopping_patience: 提前停止的耐心值
            use_tensorboard: 是否使用TensorBoard
            enable_contrastive: 是否启用对比学习
            enable_augmentation: 是否启用数据增强
            collect_features: 是否在训练后收集特征用于无监督检测
            fusion_method: 异常分数融合方法，可选 'dynamic_weight', 'static_weight', 'max', 'mean'
            
        返回:
            model_path: 保存的模型路径
        """
        # 加载训练数据
        train_logs = self.load_logs(train_file)
        logging.info(f"加载训练数据: {len(train_logs)} 条日志")
        
        # 准备窗口数据
        train_window_texts = self.prepare_log_windows(train_logs, window_type='fixed')
        logging.info(f"准备窗口数据: {len(train_window_texts)} 个窗口")
        
        # 创建数据集
        if enable_contrastive:
            train_dataset = ContrastiveLogDataset(
                self.tokenizer, train_window_texts, 
                enable_augmentation=enable_augmentation
            )
            logging.info(f"使用对比学习数据集 (启用增强: {enable_augmentation})")
        else:
            train_dataset = LogDataset(self.tokenizer, train_window_texts)
            logging.info("使用标准MLM数据集")
        
        # 加载评估数据（如果有）
        eval_dataset = None
        if eval_file and os.path.exists(eval_file):
            eval_logs = self.load_logs(eval_file)
            eval_window_texts = self.prepare_log_windows(eval_logs, window_type='fixed')
            eval_dataset = LogDataset(self.tokenizer, eval_window_texts)
            logging.info(f"加载评估数据: {len(eval_logs)} 条日志，{len(eval_window_texts)} 个窗口")
        
        # 创建模型
        self.model = create_tiny_log_bert()
        self.model.to(self.device)
        
        # 配置异常检测方法和融合器
        if self.detection_method == 'ensemble':
            logging.info(f"启用异常分数融合，使用 {fusion_method} 方法")
            # 配置融合器
            self.model.anomaly_ensemble = AnomalyScoreEnsemble(
                num_detectors=5,  # KNN, Cluster, LOF, IForest, Reconstruction
                fusion_method=fusion_method
            )
            self.model.enable_ensemble = True
        else:
            # 使用单一检测方法
            self.model.set_detection_method(self.detection_method)
            self.model.enable_ensemble = False
            logging.info(f"使用单一无监督检测方法: {self.detection_method}")
        
        # 配置数据校对器
        if enable_contrastive:
            data_collator = self.ContrastiveMLMCollator(
                tokenizer=self.tokenizer,
                mlm_probability=0.15,
                enable_augmentation=enable_augmentation
            )
            logging.info("使用对比学习数据校对器")
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=0.15
            )
            logging.info("使用标准MLM数据校对器")
        
        # 设置TensorBoard日志目录
        tensorboard_log_dir = self.tensorboard_dir if use_tensorboard else None
        
        # 配置训练参数
        training_args = TrainingArguments(
            output_dir=self.model_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            save_steps=save_steps,
            save_total_limit=3,
            logging_dir=tensorboard_log_dir,
            logging_steps=100,
            # 保存最佳模型相关配置
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,  # 对于loss，较小值更好
            # TensorBoard相关配置
            report_to=["tensorboard"] if use_tensorboard else [],
        )
        
        # 设置回调函数
        callbacks = []
        
        # 如果使用TensorBoard，添加TensorBoard回调
        if use_tensorboard:
            callbacks.append(TensorBoardCallback())
        
        # 如果有评估数据集，添加早停回调
        if eval_dataset is not None and early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_threshold=0.001
                )
            )
            logging.info(f"启用早停策略，耐心值: {early_stopping_patience}")
        
        # 创建自定义Trainer类，以支持MoCo风格的对比学习
        class ContrastiveTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                # **kwargs 仅避免报错
                # 处理MLM部分
                labels = inputs.pop("labels", None)
                
                # 对比学习相关参数
                augment_batch = inputs.pop("augment_batch", None)
                
                # 调用模型的forward方法
                outputs = model(
                    **inputs, 
                    labels=labels, 
                    augment_batch=augment_batch is not None and torch.any(augment_batch),
                    tokenizer=self.tokenizer if augment_batch is not None and torch.any(augment_batch) else None,
                    training_phase=True  # 标记处于训练阶段
                )
                
                # 获取损失
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                return (loss, outputs) if return_outputs else loss
        
        # 创建Trainer
        if enable_contrastive:
            trainer_cls = ContrastiveTrainer
            logging.info("使用对比学习Trainer")
        else:
            trainer_cls = Trainer
            logging.info("使用标准Trainer")
            
        trainer = trainer_cls(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks if callbacks else None,
            tokenizer=self.tokenizer,  # 确保tokenizer可用于数据增强
        )
        
        # 开始训练
        logging.info("开始训练模型...")
        train_result = trainer.train()
        
        # 打印训练结果
        logging.info(f"训练完成!")
        logging.info(f"训练步数: {train_result.global_step}")
        logging.info(f"训练损失: {train_result.training_loss:.4f}")
        
        # 进行最终评估
        if eval_dataset is not None:
            eval_result = trainer.evaluate()
            logging.info(f"最终评估结果:")
            for key, value in eval_result.items():
                logging.info(f"  {key}: {value:.4f}")
        
        # 保存最佳模型
        if eval_dataset is not None and training_args.load_best_model_at_end:
            logging.info("保存最佳模型...")
        else:
            logging.info("保存最终模型...")
        
        # 为无监督检测器收集特征
        if collect_features:
            self._collect_features_for_detector(train_dataset, batch_size)
        
        # 保存模型
        self.model.save_pretrained(self.model_dir)
        logging.info(f"模型已保存到: {self.model_dir}")
        
        # 同时保存tokenizer
        self.tokenizer.save_pretrained(os.path.join(self.model_dir, "tokenizer"))
    
    def _collect_features_for_detector(self, train_dataset, batch_size=32):
        """收集特征用于拟合无监督检测器"""
        logging.info("开始收集特征用于拟合无监督检测器...")
        
        # 确保模型在评估模式
        self.model.eval()
        
        # 如果检测方法是ensemble，需要为所有单一方法收集特征
        detector_methods = []
        if self.detection_method == 'ensemble':
            # 使用所有单一检测方法，排除ensemble
            detector_methods = [m for m in self.DETECTION_METHODS if m != 'ensemble' and m != 'reconstruction']
            logging.info(f"ensemble模式下将为以下检测器收集特征: {', '.join(detector_methods)}")
        else:
            # 只为当前方法收集特征
            detector_methods = [self.detection_method] if self.detection_method != 'reconstruction' else []
        
        # 如果没有需要收集特征的检测器，直接返回
        if not detector_methods:
            logging.info(f"检测方法 {self.detection_method} 不需要收集特征")
            return
            
        # 创建数据加载器
        from torch.utils.data import DataLoader
        try:
            from tqdm import tqdm
        except ImportError:
            # 如果没有tqdm，使用简单的计数器
            tqdm = lambda x, desc: x
        
        # 准备数据整理器，用于批处理
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        
        dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=data_collator
        )
        
        # 限制收集的样本数，避免内存问题
        max_samples = 10000
        collected_count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="收集特征"):
                if collected_count >= max_samples:
                    break
                
                # 处理批次数据
                # 只保留模型需要的输入
                model_inputs = {k: v.to(self.device) for k, v in batch.items() 
                              if k in ['input_ids', 'attention_mask']}
                
                # 获取特征
                outputs = self.model(
                    **model_inputs,
                    training_phase=False,  # 标记为非训练阶段
                    update_memory=True  # 更新记忆库
                )
                
                # 更新计数
                batch_size = outputs['cls_embedding'].size(0)
                collected_count += batch_size
                
                if collected_count % 1000 == 0:
                    logging.info(f"已收集 {collected_count} 个样本特征")
        
        logging.info(f"共收集了 {collected_count} 个样本特征")
        
        # 拟合检测器
        for method in detector_methods:
            if method not in self.model.anomaly_methods:
                logging.warning(f"检测方法 {method} 不在模型的anomaly_methods中，跳过")
                continue
                
            detector = self.model.anomaly_methods[method]
            success = detector.fit(force=True)
            
            if success:
                logging.info(f"成功使用 {collected_count} 个样本特征拟合 {method} 检测器")
            else:
                logging.warning(f"拟合 {method} 检测器失败，异常检测可能不准确")
    
    def evaluate(self, test_file, model_dir=None, threshold=None, eval_methods=None, eval_ensemble=True, no_labels=True):
        """
        评估模型性能
        
        参数:
            test_file: 测试数据文件（包含标签）
            model_path: 模型路径
            threshold: 异常阈值
            eval_methods: 要评估的检测方法列表，默认为当前方法
            eval_ensemble: 是否评估融合模式
            
        返回:
            results: 评估结果字典
        """
        # 加载模型
        if model_dir and os.path.exists(model_dir):
            self.model = create_tiny_log_bert(model_dir)
            self.model.to(self.device)
            self.model.eval()
        elif self.model is None:
            raise ValueError("未提供有效的模型路径，且当前没有训练好的模型")
        
        # 如果启用融合评估，添加到评估方法中
        if eval_ensemble and 'ensemble' not in (eval_methods or []):
            if eval_methods is None:
                eval_methods = [self.detection_method, 'ensemble'] if self.detection_method != 'ensemble' else ['ensemble']
            else:
                eval_methods = list(eval_methods) + ['ensemble']
            
        # 如果未指定评估方法，使用当前方法
        if eval_methods is None:
            eval_methods = [self.detection_method]
            
        # 确保所有方法都有效
        valid_methods = [method for method in eval_methods if method in self.DETECTION_METHODS]
        if len(valid_methods) < len(eval_methods):
            logging.warning(f"部分指定的方法无效，有效方法: {valid_methods}")
        eval_methods = valid_methods
        
        if not no_labels:
            # 加载测试数据（格式：{"text": "日志内容", "label": 0或1}）
            test_data = []
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        test_data.append(item)
                    except:
                        continue
            
            logging.info(f"加载测试数据: {len(test_data)} 条")
            
            # 准备数据
            texts = [item['text'] for item in test_data]
            labels = [item['label'] for item in test_data]
            
            # 评估结果字典
            all_results = {}
            
            # 获取预测特征和各种分数
            features = []
            method_scores = {method: [] for method in eval_methods}
            contrastive_distances = []
            reconstruction_errors = []
            all_method_details = {}  # 存储每个样本所有方法的分数
            
            for text in texts:
                # 单条日志评分
                log_tokens = self.window.log_tokenizer.tokenize(text)
                input_ids = log_tokens['input_ids'].to(self.device)
                attention_mask = log_tokens['attention_mask'].to(self.device)
                
                # 样本所有方法的分数
                sample_method_scores = {}
                sample_method_weights = None
                
                # 对每个评估方法进行评分
                for i, method in enumerate(eval_methods):
                    if method == 'ensemble':
                        # 确保模型启用融合模式
                        self.model.enable_ensemble = True
                    else:
                        # 使用指定的单一方法
                        self.model.enable_ensemble = False
                        self.model.set_detection_method(method)
                    
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask,
                            training_phase=False,
                            update_memory=False
                        )
                        
                        # 收集相关分数和特征
                        method_scores[method].append(outputs['anomaly_score'].item())
                        
                        # 只在第一个方法时收集通用特征
                        if i == 0:
                            features.append(outputs['cls_embedding'].cpu().numpy())
                            contrastive_distances.append(outputs['contrastive_distances'].item())
                            reconstruction_errors.append(outputs['reconstruction_error'].item())
                        
                        # 收集该样本所有方法的分数
                        if 'method_scores' in outputs:
                            for method_name, score in outputs['method_scores'].items():
                                if method_name not in sample_method_scores:
                                    sample_method_scores[method_name] = []
                                # 只保存单个样本的第一个分数（压缩维度）
                                if isinstance(score, torch.Tensor) and score.numel() > 0:
                                    sample_method_scores[method_name].append(score[0].item())
                                else:
                                    sample_method_scores[method_name].append(0.0)
                        
                        # 收集融合权重
                        if method == 'ensemble' and 'method_weights' in outputs:
                            sample_method_weights = outputs['method_weights'].cpu().numpy().tolist()
                
                # 保存样本的所有方法详情
                all_method_details[len(all_method_details)] = {
                    'method_scores': sample_method_scores,
                    'method_weights': sample_method_weights
                }
            
            # 记录各种评估结果
            for method in eval_methods:
                scores = method_scores[method]
                
                # 寻找最佳阈值（如果未提供）
                if threshold is None:
                    method_threshold, f1 = find_best_threshold(scores, labels)
                    logging.info(f"方法 {method} 最佳阈值: {method_threshold:.4f}, F1: {f1:.4f}")
                else:
                    method_threshold = threshold
                
                # 评估性能
                auc = evaluate_scores(scores, labels, model_name=f"tinylogbert_{method}")
                logging.info(f"方法 {method} AUC: {auc:.4f}")
                
                # 可视化结果（只对主方法可视化）
                if method == self.detection_method:
                    visualize_results(scores, labels, np.vstack(features), model_name=f"tinylogbert_{method}")
                
                # 计算预测结果
                predictions = [1 if s > method_threshold else 0 for s in scores]
                accuracy = sum([1 for l, p in zip(labels, predictions) if l == p]) / len(labels)
                
                # 保存方法结果
                all_results[method] = {
                    "auc": auc,
                    "threshold": method_threshold,
                    "accuracy": accuracy,
                    "predictions": predictions,
                    "scores": scores
                }
                
                # 如果是融合方法，记录各检测器的贡献情况
                if method == 'ensemble':
                    # 从模型中获取融合器的性能汇总
                    ensemble_summary = self.model.anomaly_ensemble.get_performance_summary()
                    all_results[method]['ensemble_summary'] = ensemble_summary
            
            # 评估对比距离和重构误差
            if any(d > 0 for d in contrastive_distances):
                contrastive_auc = evaluate_scores(contrastive_distances, labels, model_name="contrastive_distances")
                logging.info(f"对比距离 AUC: {contrastive_auc:.4f}")
                all_results["contrastive"] = {"auc": contrastive_auc}
            
            if any(e > 0 for e in reconstruction_errors):
                recon_auc = evaluate_scores(reconstruction_errors, labels, model_name="reconstruction_errors")
                logging.info(f"重构误差 AUC: {recon_auc:.4f}")
                all_results["reconstruction_raw"] = {"auc": recon_auc}
            
            # 输出预测结果
            main_method = self.detection_method
            # 如果主方法不在评估结果中，使用第一个方法
            if main_method not in all_results and eval_methods:
                main_method = eval_methods[0]
            
            main_predictions = all_results[main_method]["predictions"]
            main_scores = all_results[main_method]["scores"]
            
            with open(os.path.join(self.output_dir, "predictions.json"), 'w') as f:
                for i, (text, label, score, pred, c_dist, r_err) in enumerate(zip(
                    texts, labels, main_scores, main_predictions, 
                    contrastive_distances, reconstruction_errors
                )):
                    result = {
                        "id": i,
                        "text": text,
                        "true_label": label,
                        "score": float(score),
                        "contrastive_distance": float(c_dist),
                        "reconstruction_error": float(r_err),
                        "predicted_label": pred,
                        "correct": label == pred,
                        # 添加所有方法的分数和细节
                        "method_scores": {m: float(method_scores[m][i]) for m in eval_methods},
                        "method_details": all_method_details.get(i, {})
                    }
                    f.write(json.dumps(result) + '\n')
            
            # 添加通用结果信息
            main_results = all_results[main_method]
            main_results.update({
                "num_samples": len(texts),
                "num_anomalies": sum(labels),
                "detection_method": main_method,
                "all_methods": {m: all_results[m]["auc"] for m in eval_methods}
            })
            
            return main_results
    
        else:
            logging.warning("NotImplementedError: 待实现无监督方法")
            return {}

    def detect(self, log_text, model_dir=None, threshold=0.5, method=None):
        """
        检测单条日志是否异常
        
        参数:
            log_text: 日志文本
            model_path: 模型路径
            threshold: 异常阈值
            method: 检测方法
            
        返回:
            result: 检测结果字典
        """
        # 加载模型（如果需要）
        if self.model is None and model_dir:
            self.model = create_tiny_log_bert(model_dir)
            self.model.to(self.device)
            self.model.eval()
        elif self.model is None:
            raise ValueError("未提供有效的模型路径，且当前没有加载模型")
        
        # 设置检测方法
        if method is not None:
            if method == 'ensemble':
                self.model.enable_ensemble = True
            else:
                self.model.enable_ensemble = False
                if method in self.DETECTION_METHODS:
                    self.model.set_detection_method(method)
                else:
                    method = self.detection_method
        else:
            method = self.detection_method
            if method == 'ensemble':
                self.model.enable_ensemble = True
            else:
                self.model.enable_ensemble = False
                self.model.set_detection_method(method)
        
        # 对日志进行异常评分
        log_tokens = self.window.log_tokenizer.tokenize(log_text)
        input_ids = log_tokens['input_ids'].to(self.device)
        attention_mask = log_tokens['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                training_phase=False
            )
            
            score = outputs['anomaly_score'].item()
            contrastive_distance = outputs.get('contrastive_distances', torch.tensor([0.0])).item()
            reconstruction_error = outputs.get('reconstruction_error', torch.tensor([0.0])).item()
            
            # 获取各方法的分数和权重
            method_scores = {}
            method_weights = []
            
            if 'method_scores' in outputs:
                for method_name, score_tensor in outputs['method_scores'].items():
                    if isinstance(score_tensor, torch.Tensor) and score_tensor.numel() > 0:
                        method_scores[method_name] = score_tensor[0].item()
                    else:
                        method_scores[method_name] = 0.0
                        
            if 'method_weights' in outputs:
                method_weights = outputs['method_weights'].cpu().numpy().tolist()
        
        # 判断是否异常
        is_anomaly = score > threshold
        
        # 返回结果
        return {
            "log": log_text,
            "score": float(score),
            "contrastive_distance": float(contrastive_distance),
            "reconstruction_error": float(reconstruction_error),
            "threshold": threshold,
            "is_anomaly": bool(is_anomaly),
            "detection_method": method,
            "method_scores": method_scores,
            "method_weights": method_weights
        }
    
    def detect_sequence(self, log_list, model_dir=None, window_type='sliding', stride=1, threshold=0.5, method=None):
        """
        检测日志序列中的异常
        
        参数:
            log_list: 日志文本列表
            model_path: 模型路径
            window_type: 窗口类型 ('fixed' 或 'sliding')
            stride: 滑动窗口的步长
            threshold: 异常阈值
            method: 检测方法
            
        返回:
            results: 检测结果列表
        """
        # 加载模型（如果需要）
        if self.model is None and model_dir:
            self.model = create_tiny_log_bert(model_dir)
            self.model.to(self.device)
            self.model.eval()
        elif self.model is None:
            raise ValueError("未提供有效的模型路径，且当前没有加载模型")
        
        # 设置检测方法
        if method is not None:
            if method == 'ensemble':
                self.model.enable_ensemble = True
            else:
                self.model.enable_ensemble = False
                if method in self.DETECTION_METHODS:
                    self.model.set_detection_method(method)
                else:
                    method = self.detection_method
        else:
            method = self.detection_method
            if method == 'ensemble':
                self.model.enable_ensemble = True
            else:
                self.model.enable_ensemble = False
                self.model.set_detection_method(method)
        
        # 准备窗口
        if window_type == 'fixed':
            window_tokens, _ = self.window.create_fixed_windows(log_list)
        else:  # sliding
            window_tokens = self.window.create_sliding_windows(log_list, stride)
        
        if not window_tokens:
            return []
        
        # 批量处理
        batch = self.window.batch_windows(window_tokens)
        
        if batch is None:
            return []
        
        # 移到device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # 获取分数
        window_scores = []
        contrastive_distances = []
        reconstruction_errors = []
        window_results = []
        all_method_scores = []
        all_method_weights = []
        
        with torch.no_grad():
            for i in range(batch['input_ids'].size(0)):
                input_ids = batch['input_ids'][i:i+1]
                attention_mask = batch['attention_mask'][i:i+1]
                
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    training_phase=False
                )
                
                score = outputs['anomaly_score'].item()
                contrastive_distance = outputs.get('contrastive_distances', torch.tensor([0.0])).item()
                reconstruction_error = outputs.get('reconstruction_error', torch.tensor([0.0])).item()
                is_anomaly = score > threshold
                
                window_scores.append(score)
                contrastive_distances.append(contrastive_distance)
                reconstruction_errors.append(reconstruction_error)
                
                # 获取各方法的分数和权重
                method_scores = {}
                method_weights = []
                
                if 'method_scores' in outputs:
                    for method_name, score_tensor in outputs['method_scores'].items():
                        if isinstance(score_tensor, torch.Tensor) and score_tensor.numel() > 0:
                            method_scores[method_name] = score_tensor[0].item()
                        else:
                            method_scores[method_name] = 0.0
                            
                if 'method_weights' in outputs:
                    method_weights = outputs['method_weights'].cpu().numpy().tolist()
                
                all_method_scores.append(method_scores)
                all_method_weights.append(method_weights)
                
                # 对于滑动窗口，记录窗口开始位置
                start_idx = i * stride if window_type == 'sliding' else i * self.window_size
                end_idx = start_idx + self.window_size
                
                window_logs = log_list[start_idx:end_idx]
                window_text = " [SEP] ".join(window_logs)
                
                window_results.append({
                    "window_idx": i,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "logs": window_logs,
                    "score": float(score),
                    "contrastive_distance": float(contrastive_distance),
                    "reconstruction_error": float(reconstruction_error),
                    "is_anomaly": bool(is_anomaly),
                    "method_scores": method_scores,
                    "method_weights": method_weights
                })
        
        # 计算整体统计信息
        avg_score = np.mean(window_scores) if window_scores else 0.0
        max_score = np.max(window_scores) if window_scores else 0.0
        avg_contrastive = np.mean(contrastive_distances) if contrastive_distances else 0.0
        avg_reconstruction = np.mean(reconstruction_errors) if reconstruction_errors else 0.0
        anomaly_windows = [r for r in window_results if r['is_anomaly']]
        
        # 如果是融合方法，获取融合器性能摘要
        ensemble_summary = None
        if method == 'ensemble':
            ensemble_summary = self.model.anomaly_ensemble.get_performance_summary()
        
        # 添加整体结果
        overall_result = {
            "num_windows": len(window_results),
            "avg_score": float(avg_score),
            "max_score": float(max_score),
            "avg_contrastive_distance": float(avg_contrastive),
            "avg_reconstruction_error": float(avg_reconstruction),
            "num_anomaly_windows": len(anomaly_windows),
            "anomaly_ratio": float(len(anomaly_windows) / len(window_results)) if window_results else 0.0,
            "detection_method": method,
            "windows": window_results,
            "ensemble_summary": ensemble_summary
        }
        
        return overall_result 