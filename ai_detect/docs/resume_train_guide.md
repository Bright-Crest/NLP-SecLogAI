# AI日志异常检测模型训练恢复指南

本文档说明如何使用`resume_train.py`脚本来恢复已中断的AI日志异常检测模型训练过程。

## 基本介绍

在长时间训练日志异常检测模型时，可能因各种原因（如电源中断、系统崩溃、网络断开等）导致训练过程被迫中断。为了不浪费已经投入的计算资源和时间，`resume_train.py`脚本提供了从断点恢复训练的功能。

这个脚本能够：
1. 自动查找最新的检查点或指定的检查点
2. 从检查点恢复模型状态、优化器状态和训练进度
3. 继续完成剩余的训练流程
4. 保存训练完成的模型并进行评估

## 使用方法

### 基本用法

```bash
python ai_detect/resume_train.py --train_file <训练数据文件> --eval_file <评估数据文件> --checkpoint_dir <检查点目录>
```

这个命令会自动查找检查点目录中最新的检查点，并从该检查点恢复训练。

### 常用参数

以下是一些常用的命令行参数：

| 参数 | 说明 | 默认值 |
| ---- | ---- | ---- |
| `--train_file` | 训练数据文件路径（必需） | 无 |
| `--eval_file` | 评估数据文件路径（可选） | 无 |
| `--checkpoint_dir` | 检查点目录，包含之前训练的模型 | `./ai_detect/checkpoint` |
| `--output_dir` | 输出结果目录 | `./ai_detect/output` |
| `--checkpoint_path` | 指定要恢复的检查点路径 | 无（自动选择最新检查点） |
| `--use_best_checkpoint` | 使用验证损失最小的检查点而不是最新的检查点 | 否 |
| `--num_epochs` | 继续训练的轮次 | 5 |
| `--batch_size` | 批次大小 | 128 |
| `--not_early_stopping` | 禁用早停机制 | 否（默认启用） |
| `--detection_method` | 无监督异常检测方法 | `ensemble` |

### 示例

1. **从最新检查点恢复训练，训练10轮：**

```bash
python ai_detect/resume_train.py --train_file ./dataset/train.log --eval_file ./dataset/eval.log --num_epochs 10
```

2. **从特定检查点恢复训练：**

```bash
python ai_detect/resume_train.py --train_file ./dataset/train.log --checkpoint_path ./ai_detect/checkpoint/checkpoint-1000
```

3. **使用验证损失最小的检查点恢复训练：**

```bash
python ai_detect/resume_train.py --train_file ./dataset/train.log --eval_file ./dataset/eval.log --use_best_checkpoint
```

4. **使用不同的异常检测方法恢复训练：**

```bash
python ai_detect/resume_train.py --train_file ./dataset/train.log --detection_method knn
```

## 高级用法

### 更改模型输出目录

如果希望将恢复训练后的模型保存到新的目录，可以使用`--model_dir`参数：

```bash
python ai_detect/resume_train.py --train_file ./dataset/train.log --model_dir ./ai_detect/new_model
```

### 调整早停设置

默认情况下，如果提供了评估数据，脚本会启用早停机制，在连续3次评估没有改进后停止训练。您可以通过`--patience`参数调整耐心值：

```bash
python ai_detect/resume_train.py --train_file ./dataset/train.log --eval_file ./dataset/eval.log --patience 5
```

或者完全禁用早停：

```bash
python ai_detect/resume_train.py --train_file ./dataset/train.log --eval_file ./dataset/eval.log --not_early_stopping
```

### 禁用对比学习或数据增强

如果原始训练使用了对比学习或数据增强，但您想在恢复训练时禁用这些功能：

```bash
python ai_detect/resume_train.py --train_file ./dataset/train.log --not_use_contrastive --not_use_augmentation
```

## 故障排除

### 找不到检查点

如果脚本报告找不到检查点，请检查：
1. 检查点目录路径是否正确
2. 检查点目录中是否存在`checkpoint-*`子目录
3. 模型是否曾经成功保存过检查点（训练步数是否足够）

### 恢复训练失败

如果恢复训练过程中出现错误，常见原因包括：
1. 检查点文件损坏或不完整
2. GPU内存不足
3. 原始训练和恢复训练使用的模型架构不同

请尝试：
1. 使用较早的检查点
2. 减小批次大小
3. 确保使用与原始训练相同的模型架构和超参数

## 注意事项

1. 恢复训练时，建议使用与原始训练相同的训练数据和超参数，以减少不一致性引起的问题
2. 请确保有足够的磁盘空间存储检查点和新的模型文件
3. 对于大型日志数据集，训练过程可能需要较长时间，建议在稳定的环境中运行 