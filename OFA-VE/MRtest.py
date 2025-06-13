import os
import shutil
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

ofa_pipe = pipeline(Tasks.visual_entailment, model='damo/ofa_visual-entailment_snli-ve_large_en')

tuple_text_dir ='data/entailonly'
img_path = 'dt/img'

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

    img_prefix = txt_file.split('_')[0]
    img_file = img_prefix + '.jpg'
    img = os.path.join(img_path, img_file)  # Full path to the image
    print("测试img路径:" + img)

    with open(tuple_text_path, 'r', encoding='utf-8') as f:
        txt = ''
        for line in f:
            text = line.strip()
            # 跳过空行
            if not text:
                continue
            print(text)
            input = {'image': img, 'text': text}  # Prepare input for the model
            result = ofa_pipe(input)
            if OutputKeys.LABELS in result:
                labels = result[OutputKeys.LABELS]
                # 处理列表中的情况，这里假设结果只有一个标签
                if isinstance(labels, list) and len(labels) > 0:
                    label = labels[0]
                    # 将标签映射到新的值
                    if label == 'yes':
                        ve_result = 'Entailment'
                    elif label == 'no':
                        ve_result = 'Contradiction'
                    elif label == 'maybe':
                        ve_result = 'Neutral'
            print(ve_result)
            txt += f"{text} ||{ve_result}\n"
    with open(tuple_text_path, 'w', encoding='utf-8') as f:
        f.write(txt)