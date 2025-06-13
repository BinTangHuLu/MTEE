import os
import shutil

img_text_dir = 'data/img_text'
dataset_name = os.path.dirname(img_text_dir)#数据集的路径 如：dataset/VOC2007
dataset_img=os.path.basename(img_text_dir)#数据集下的图片文件夹名称 如：JPEGImages

result_dir = dataset_name + '/results' # 原结果路径
texts_result_dir = 'data/entailonly'
gold_label_dir = 'data/gold_label'

txt_files = [f for f in os.listdir(texts_result_dir) if f.lower().endswith('.txt')]

pass_num = 0
fail_num = 0
# 一张图片可以有多个描述，所以先找txt
for txt_file in txt_files:
    txt_path = os.path.join(result_dir, txt_file)
    gold_label_path = os.path.join(gold_label_dir, txt_file)
    with open(gold_label_path, 'r', encoding='utf-8') as f:
        gold_label = f.read().strip()
    print('-' * 100)
    print('原始text结果路径：', txt_path)  # 原txt路径
    text_path = os.path.join(img_text_dir, txt_file)
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
        print(text)
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
        if 'contradiction' in mr_result or 'neutral' in mr_result:
            fail_num += 1
            print('fail')
        else:
            pass_num += 1
            print('pass')

print('测试用例总数：', pass_num+fail_num)
print('测试通过个数：', pass_num)
print('测试失败个数：', fail_num)