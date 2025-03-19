import re

def extract_loss_values(file_path):
    # 定义正则表达式匹配 loss 和 eval_loss
    loss_pattern = re.compile(r"'loss': ([\d\.]+)")
    eval_loss_pattern = re.compile(r"'eval_loss': ([\d\.]+)")

    # 用于存储提取的值
    loss_values = []
    eval_loss_values = []

    # 打开文件并逐行读取
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 匹配 loss
            loss_match = loss_pattern.search(line)
            if loss_match:
                loss_values.append(float(loss_match.group(1)))

            # 匹配 eval_loss
            eval_loss_match = eval_loss_pattern.search(line)
            if eval_loss_match:
                eval_loss_values.append(float(eval_loss_match.group(1)))

    return loss_values, eval_loss_values

# 示例文件路径
file_path = './torch3090log.txt'  # 替换为你的文件路径

# 提取 loss 和 eval_loss
loss_values, eval_loss_values = extract_loss_values(file_path)

# 输出结果
print("Loss values:", loss_values)
print("Eval loss values:", eval_loss_values)