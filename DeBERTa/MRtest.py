import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

deberta_path = 'nli-deberta-v3-base'
model = AutoModelForSequenceClassification.from_pretrained(deberta_path)
tokenizer = AutoTokenizer.from_pretrained(deberta_path)
model.eval()

tuple_text_dir ='data/entailonly'
premise_dir = 'data/premise'


txt_files = [f for f in os.listdir(tuple_text_dir) if f.lower().endswith('.txt')]
for txt_file in txt_files:
    tuple_text_path = os.path.join(tuple_text_dir, txt_file)
    print('-' * 100)
    print('衍生测试用例句子路径：', tuple_text_path)  # 衍生测试用例句子路径

    skip_file = False
    with open(tuple_text_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '||' in line:
                skip_file = True
                break  # 直接跳出当前文件的循环，避免继续处理

    if skip_file:
        print(f"文件 {txt_file} 已经推理过，跳过处理...")
        continue  # 跳过当前文件的处理

    img_id = txt_file.split('_')[0]
    s1_index = txt_file.split('_')[1]
    s2_index = txt_file.split('_')[2]
    premise_file = os.path.join(premise_dir, f"{img_id}_{s1_index}.txt")

    with open(premise_file, 'r', encoding='utf-8') as f:
        premise_text = f.read().strip()
        print('premise_text:', premise_text)
    with open(tuple_text_path, 'r', encoding='utf-8') as f:
        txt = ''
        for line in f:
            text = line.strip()
            if not text:
                continue
            print(text)
            features = tokenizer(premise_text, text, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                scores = model(**features).logits
                label_mapping = ['contradiction', 'entailment', 'neutral']
                labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
                inference_result = labels[0]
                print('result：', inference_result)
            txt += f"{text} ||{inference_result}\n"
    with open(tuple_text_path, 'w', encoding='utf-8') as f:
        f.write(txt)