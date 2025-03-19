# 安装依赖（如有需要）
# pip install transformers torch accelerate datasets evaluate
# pip install scikit-learn
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    BitForImageClassification,
    AutoImageProcessor,
    Trainer,
    TrainingArguments
)

# 1. 数据集选择与说明
# Oxford Flowers-102数据集：
# - 包含102种英国常见花卉
# - 8,189张高质量图像（训练：1,020，验证：1,020，测试：6,149）
# - 图像分辨率从500x500到多种尺寸
# 选择理由：
# 1. 中等规模适合端到端微调（非冻结训练）
# 2. 细粒度分类任务展示模型强大特征提取能力
# 3. 实际应用场景广泛（植物识别、生态研究等）
# 测试函数定义
def test_single_image(model, processor, image, device='cuda'):
    # 预处理图像
    inputs = processor(
        images=image.convert('RGB'),
        return_tensors="pt",
        size={"height": 384, "width": 384}
    )
    
    # 将输入移到GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    model.eval()
    
    # 进行预测
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(-1)
    
    return predictions.item()

# 只加载训练集并进行拆分
dataset = load_dataset("dpdl-benchmark/oxford_flowers102", split="train")
# 将训练集按8:2的比例拆分为训练集和测试集
dataset = dataset.train_test_split(test_size=0.2, seed=42)
print(dataset)



# 2. 加载预处理器和模型
model_name = "HorcruxNo13/bit-50"  # 使用更大的bit模型
processor = AutoImageProcessor.from_pretrained(model_name)
model = BitForImageClassification.from_pretrained(
    model_name,
    num_labels=102,
    ignore_mismatched_sizes=True
)
# 选择一个测试集样本进行测试
test_image = dataset['test'][0]['image']
test_label = dataset['test'][0]['label']

print("\n=== 训练前测试 ===")
pred_before = test_single_image(model, processor, test_image)
print(f"真实标签: {test_label}")
print(f"预测标签: {pred_before}")

# 3. 数据集预处理
def transform(examples):
    # 统一转换RGB格式
    images = [img.convert("RGB") for img in examples["image"]]
    # 使用BiT推荐尺寸384x384（提升精度同时适应显存）
    inputs = processor(
        images=images,
        return_tensors="pt",
        size={"height": 384, "width": 384}
    )
    # 添加标签映射
    inputs["labels"] = examples["label"]
    return inputs

# 应用预处理并设置格式
encoded_dataset = dataset.map(
    transform,
    batched=True,
    remove_columns=["image", "label"]
)
encoded_dataset.set_format("torch")

# 4. 训练配置（修改学习率和训练轮数）
training_args = TrainingArguments(
    output_dir="./torch_bit_flowers102",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,  # 提高学习率从3e-5到5e-5
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    num_train_epochs=10,  # 增加训练轮数从5轮到10轮
    fp16=True,
    gradient_accumulation_steps=1,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    optim="adamw_torch",
    remove_unused_columns=False
)

# 5. 定义评估指标
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 6. 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],  # 原来是 validation
    compute_metrics=compute_metrics,
)

# 7. 训练与评估
trainer.train()

# 最终测试集评估
test_results = trainer.evaluate()  # 直接使用拆分出的测试集进行评估
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")

print("\n=== 训练后测试 ===")
pred_after = test_single_image(model, processor, test_image)
print(f"真实标签: {test_label}")
print(f"预测标签: {pred_after}")
