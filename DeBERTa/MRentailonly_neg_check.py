import os
import shutil

texts_result_dir = 'data/entailonly_neg'
premise_dir = 'data/premise'
hypothesis_dir = 'data/hypothesis'
result_dir = 'data/result'
Merge_hypothesis_dir = 'data/Merge_hypothesis'
gold_label_dir = 'data/gold_label'

txt_files = [f for f in os.listdir(texts_result_dir) if f.lower().endswith('.txt')]

pass_num = 0
fail_num = 0

for txt_file in txt_files:
    txt_path = os.path.join(result_dir, txt_file)
    gold_label_path = os.path.join(gold_label_dir, txt_file)
    with open(gold_label_path, 'r', encoding='utf-8') as f:
        gold_label = f.read().strip()
    print('-' * 100)
    print('原始结果路径：', txt_path)  # 原txt路径

    img_id = txt_file.split('_')[0]
    s1_index = txt_file.split('_')[1]
    s2_index = txt_file.split('_')[2]
    premise_text_path = os.path.join(premise_dir, f"{img_id}_{s1_index}.txt")

    with open(premise_text_path, 'r', encoding='utf-8') as f:
        premise_text = f.read()
        print("premise:", premise_text)
    hypothesis_text_path = os.path.join(hypothesis_dir, txt_file)
    with open(hypothesis_text_path, 'r', encoding='utf-8') as f:
        hypothesis_text = f.read()
        print("hypothesis:", hypothesis_text)
    with open(txt_path, 'r', encoding='utf-8') as f:
        result = f.read().strip()
        print(result)
    mr_txt_path = os.path.join(texts_result_dir, txt_file)
    mr_result = []
    with open(mr_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            result_line = line.strip().split('||')
            mr_result.append(result_line[1])
            print(result_line[0])
    print(mr_result)

    if result == 'entailment':
        if 'entailment' in mr_result or 'neutral' in mr_result:
            fail_num += 1
            print('fail')
        else:
            pass_num += 1
            print('pass')

print('测试用例总数：', pass_num+fail_num)
print('测试通过个数：', pass_num)
print('测试失败个数：', fail_num)