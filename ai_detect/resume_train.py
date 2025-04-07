import os
import sys
import argparse
import logging
import time
import glob
import json
import types
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ai_models.anomaly_detector import AnomalyDetector, LogDataset, ContrastiveLogDataset
from app.ai_models.tinylogbert import AnomalyScoreEnsemble
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "app", "checkpoint")
OUTPUT_DIR = os.path.join(ROOT_DIR, "ai_detect", "output")


def setup_logging(output_dir):
    """设置日志配置"""
    try:
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置清晰的日志文件名
        log_file = os.path.join(output_dir, f"resume_train_{time.strftime('%Y%m%d_%H%M%S')}.log")
        
        # 尝试创建一个空的日志文件，检查写入权限
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"# 日志开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 检查文件是否被创建
        if not os.path.exists(log_file):
            print(f"警告: 无法创建日志文件 {log_file}")
            return False
            
        # 配置根日志记录器
        root_logger = logging.getLogger()
        # 重置处理程序，避免重复
        if root_logger.handlers:
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                
        # 创建并配置文件处理程序
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        
        # 创建并配置控制台处理程序
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # 添加处理程序到根日志记录器
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.INFO)
        
        # 添加初始日志条目，确认日志系统已启动
        logging.info(f"日志系统已初始化，日志文件: {log_file}")
        
        return True
    except Exception as e:
        print(f"设置日志系统时发生错误: {str(e)}")
        # 设置基本的控制台日志以便继续运行
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        logging.warning(f"无法设置文件日志，将只使用控制台日志: {str(e)}")
        return False


def find_latest_checkpoint(checkpoint_dir):
    """
    查找最新的检查点目录
    
    参数:
        checkpoint_dir: 检查点根目录
        
    返回:
        latest_checkpoint: 最新检查点目录的路径，如果没有则返回None
    """
    # 查找所有checkpoint-*目录
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        return None
    
    # 按修改时间排序
    checkpoint_dirs = sorted(checkpoint_dirs, key=os.path.getmtime, reverse=True)
    
    # 返回最新的检查点目录
    return checkpoint_dirs[0]


def get_best_checkpoint(output_dir):
    """
    获取训练过程中的最佳检查点
    
    参数:
        output_dir: 输出目录，包含训练状态和最佳模型信息
        
    返回:
        best_checkpoint: 最佳检查点路径，如果不存在则返回None
    """
    # 查找trainer_state.json文件
    trainer_state_file = os.path.join(output_dir, "trainer_state.json")
    
    if not os.path.exists(trainer_state_file):
        return None
    
    try:
        with open(trainer_state_file, 'r', encoding='utf-8') as f:
            trainer_state = json.load(f)
        
        # 获取最佳检查点信息
        if 'best_model_checkpoint' in trainer_state and trainer_state['best_model_checkpoint']:
            return trainer_state['best_model_checkpoint']
    except Exception as e:
        logging.error(f"读取训练状态文件失败: {str(e)}")
    
    return None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="恢复训练日志异常检测模型")
    
    # 基本参数
    parser.add_argument("--train_file", type=str, required=True,
                        help="训练数据文件路径")
    parser.add_argument("--eval_file", type=str, default=None,
                        help="评估数据文件路径（可选）")
    parser.add_argument("--checkpoint_dir", type=str, default=MODEL_DIR,
                        help="检查点目录，包含之前训练的模型")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="输出结果目录")
    
    # 恢复训练特定参数
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="指定要恢复的检查点路径，默认使用最新检查点")
    parser.add_argument("--use_best_checkpoint", action="store_true",
                        help="是否使用验证损失最小的检查点而不是最新的检查点")
    
    # 模型参数
    parser.add_argument("--tokenizer_name", type=str, default="prajjwal1/bert-mini",
                        help="使用的tokenizer名称")
    parser.add_argument("--window_size", type=int, default=10,
                        help="日志窗口大小")
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="总共要训练的轮次，不是从检查点恢复后要训练的轮次")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="批次大小")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="保存模型的步数间隔")
    
    # 新增参数：早停和TensorBoard
    parser.add_argument("--not_early_stopping", action="store_true", 
                        help="是否不启用早停机制")
    parser.add_argument("--patience", type=int, default=3,
                        help="早停的耐心值（连续多少次评估没有改进后停止）")
    parser.add_argument("--not_use_tensorboard", action="store_true",
                        help="是否不使用TensorBoard记录训练过程")
    
    # 无监督异常检测方法选择
    parser.add_argument("--detection_method", type=str, default="ensemble",
                        choices=["knn", "cluster", "lof", "iforest", "reconstruction", "ensemble"],
                        help="无监督异常检测方法")
    parser.add_argument("--fusion_method", type=str, default="dynamic_weight",
                        choices=["dynamic_weight", "static_weight", "max", "mean"],
                        help="异常分数融合方法，用于ensemble检测方法")
    parser.add_argument("--eval_all_methods", action="store_true",
                        help="评估时是否测试所有检测方法")
    parser.add_argument("--not_eval_ensemble", action="store_true",
                        help="评估时是否不评估融合模式（如果使用单一检测方法）")
    parser.add_argument("--not_use_contrastive", action="store_true",
                        help="不使用对比学习")
    parser.add_argument("--not_use_augmentation", action="store_true",
                        help="不使用数据增强")
    
    return parser.parse_args()


def patch_anomaly_detector_for_resuming(detector, checkpoint_path):
    """
    通过monkey patch修改AnomalyDetector类，使其支持从检查点恢复训练
    
    参数:
        detector: AnomalyDetector实例
        checkpoint_path: 检查点路径
    """
    # 保存原始的train方法
    original_train = detector.train
    
    # 定义新的train方法，支持断点续训
    def train_with_resume(self, train_file, eval_file=None, num_epochs=10, batch_size=32, save_steps=500,
                          early_stopping_patience=3, use_tensorboard=True, enable_contrastive=True, 
                          enable_augmentation=True, collect_features=True, fusion_method='dynamic_weight'):
        """
        从检查点恢复训练模型
        
        参数:
            与原始train方法相同
            
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
            overwrite_output_dir=False,  # 不覆盖输出目录，以保留之前的检查点
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            save_steps=save_steps,
            save_total_limit=5,
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
            callbacks=callbacks if callbacks else None
        )
        
        # 开始训练 - 关键修改: 从检查点恢复训练
        logging.info(f"从检查点 {checkpoint_path} 恢复训练...")
        train_result = trainer.train(resume_from_checkpoint=checkpoint_path)
        
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
        logging.info(f"Tokenizer已保存到: {os.path.join(self.model_dir, 'tokenizer')}")
        
        return self.model_dir
    
    # 替换原始方法为新方法
    detector.train = types.MethodType(train_with_resume, detector)
    logging.info("已启用断点续训功能")
    
    return detector


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    logs_dir = os.path.join(args.output_dir, "logs")
    logging_success = setup_logging(logs_dir)
    
    if not logging_success:
        logging.warning(f"警告: 日志系统设置失败，将继续执行但日志可能不会被正确保存")
    
    # 确保输出目录存在
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"创建输出目录失败: {str(e)}")
        return
    
    # 记录参数
    logging.info("恢复训练参数:")
    for arg in vars(args):
        logging.info(f"  {arg}: {getattr(args, arg)}")
    
    # 确定恢复训练的检查点路径
    checkpoint_path = args.checkpoint_path
    
    if checkpoint_path is None:
        # 如果用户选择使用最佳检查点
        if args.use_best_checkpoint:
            checkpoint_path = get_best_checkpoint(args.checkpoint_dir)
            if checkpoint_path:
                logging.info(f"使用验证损失最小的检查点: {checkpoint_path}")
            else:
                logging.warning("未找到最佳检查点信息，将尝试使用最新检查点")
        
        # 如果未找到最佳检查点或用户未选择使用最佳检查点，则使用最新检查点
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
            if checkpoint_path:
                logging.info(f"使用最新检查点: {checkpoint_path}")
            else:
                logging.warning(f"在目录 {args.checkpoint_dir} 中未找到检查点")
                logging.info("将从初始模型开始训练")
                checkpoint_path = args.checkpoint_dir
    else:
        # 用户指定了检查点路径
        if not os.path.exists(checkpoint_path):
            logging.error(f"指定的检查点路径 {checkpoint_path} 不存在")
            return
        logging.info(f"使用指定的检查点: {checkpoint_path}")
    
    train_file = args.train_file
    eval_file = args.eval_file
    
    # 检查训练文件是否存在
    if not os.path.exists(train_file):
        logging.error(f"训练文件 {train_file} 不存在")
        return
    
    # 检查评估文件是否存在
    if eval_file and not os.path.exists(eval_file):
        logging.warning(f"评估文件 {eval_file} 不存在，将不进行评估")
        eval_file = None
            
    # 确定是否使用早停
    use_early_stopping = not args.not_early_stopping and eval_file is not None
    if use_early_stopping:
        logging.info(f"启用早停机制，耐心值: {args.patience}")
    else:
        logging.info("未启用早停机制")
    
    # 确定是否使用TensorBoard
    if not args.not_use_tensorboard:
        logging.info("启用TensorBoard记录")
        tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        logging.info(f"TensorBoard日志将保存在: {tensorboard_dir}")
        logging.info(f"启动TensorBoard: tensorboard --logdir={tensorboard_dir}")
    
    # 确定对比学习和数据增强设置
    enable_contrastive = not args.not_use_contrastive
    enable_augmentation = not args.not_use_augmentation
    
    # 初始化检测器，使用检查点路径作为模型目录
    detector = AnomalyDetector(
        model_dir=args.checkpoint_dir,  # 使用输出模型目录，而不是检查点路径
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        window_size=args.window_size,
        detection_method=args.detection_method
    )
    
    # 应用断点续训补丁
    detector = patch_anomaly_detector_for_resuming(detector, checkpoint_path)
    
    # 开始恢复训练
    logging.info("开始恢复训练模型...")
    logging.info(f"从检查点: {checkpoint_path}")
    logging.info(f"训练数据: {train_file}")
    logging.info(f"评估数据: {eval_file if eval_file else '无'}")
    logging.info(f"异常检测方法: {args.detection_method}")
    if args.detection_method == 'ensemble':
        logging.info(f"融合方法: {args.fusion_method}")
    logging.info(f"对比学习: {'启用' if enable_contrastive else '禁用'}")
    logging.info(f"数据增强: {'启用' if enable_augmentation else '禁用'}")
    
    start_time = time.time()
    
    try:
        # 恢复训练模型，注意这里我们把checkpoint_path指定为模型目录，并设置resume_from_checkpoint=True
        detector.train(
            train_file=train_file,
            eval_file=eval_file,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            save_steps=args.save_steps,
            early_stopping_patience=args.patience if use_early_stopping else 0,
            use_tensorboard=not args.not_use_tensorboard,
            enable_contrastive=enable_contrastive,
            enable_augmentation=enable_augmentation,
            fusion_method=args.fusion_method
        )
        
        # 进行评估（如果有评估数据）
        if eval_file is not None:
            logging.info("对训练好的模型进行评估...")
            
            # 决定评估哪些方法
            eval_methods = None  # 默认当前方法
            if args.eval_all_methods:
                # 使用所有单一方法，不包括ensemble
                eval_methods = [m for m in detector.DETECTION_METHODS if m != 'ensemble']
                logging.info(f"将评估所有检测方法: {', '.join(eval_methods)}")
            
            # 进行评估
            eval_results = detector.evaluate(
                test_file=eval_file, 
                model_dir=args.checkpoint_dir,
                eval_methods=eval_methods,
                eval_ensemble=not args.not_eval_ensemble,
                no_labels=True  # 使用无监督评估，即使可能有标签也以无标签方式评估
            )
            
            # 输出评估结果
            logging.info(f"无监督评估结果:")
            logging.info(f"  主方法: {eval_results.get('detection_method', args.detection_method)}")
            logging.info(f"  推荐阈值: {eval_results.get('threshold', 0.0):.4f}")
            logging.info(f"  样本数量: {eval_results.get('num_samples', 0)}")
            logging.info(f"  可能异常样本数量: {eval_results.get('num_anomalies', 0)} ({eval_results.get('anomaly_ratio', 0.0)*100:.2f}%)")
            
            # 打印模型评估等级
            if 'summary' in eval_results:
                summary = eval_results['summary']
                logging.info(f"  模型评估: {summary.get('grade', '未知')} (分数: {summary.get('score', 0)})")
                logging.info(f"  评估摘要: {summary.get('summary', '无')}")
                
                # 打印评估因素
                if 'factors' in summary and summary['factors']:
                    logging.info("  评估因素:")
                    for factor in summary['factors']:
                        logging.info(f"    - {factor}")
            
            # 打印建议
            if 'suggestions' in eval_results and eval_results['suggestions']:
                logging.info("  建议:")
                for suggestion in eval_results['suggestions']:
                    logging.info(f"    - {suggestion}")
            
            # 如果评估了多个方法，输出所有方法的质量
            if 'all_methods' in eval_results:
                logging.info("  各方法评估:")
                for method, quality in eval_results['all_methods'].items():
                    logging.info(f"    {method}: {quality}")
            
            # 如果使用了融合方法，输出权重
            if 'ensemble_weights' in eval_results:
                logging.info("  融合器权重:")
                for method, weight in eval_results['ensemble_weights'].items():
                    logging.info(f"    {method}: {weight:.4f}")
            
        logging.info(f"恢复训练耗时: {(time.time() - start_time) / 60:.2f} 分钟")
        
    except Exception as e:
        logging.error(f"恢复训练过程中发生错误: {str(e)}")
        raise
    
    logging.info("恢复训练完成!")


if __name__ == "__main__":
    main() 