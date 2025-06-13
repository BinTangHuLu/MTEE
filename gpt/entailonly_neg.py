import os
from negate import Negator
# Use a Transformer model (en_core_web_trf):
negator = Negator(use_transformers=True, fail_on_unsupported=True)


entail_only = 'entailonly_neg'
tuple = 'tuple_entailonly_neg'

txt_files = [f for f in os.listdir(entail_only) if f.lower().endswith('.txt')]

sum = 0
for txt_file in txt_files:
    source_file = os.path.join(entail_only, txt_file)
    tuple_file = os.path.join(tuple, txt_file)  # 对应的tuple文件

    # 读取entail_only和tuple文件
    with open(source_file, 'r', encoding='utf-8') as f_entail, open(tuple_file, 'r', encoding='utf-8') as f_tuple:
        entail_lines = f_entail.readlines()
        tuple_lines = f_tuple.readlines()

    # 存储更新后的内容
    updated_entail = []
    updated_tuple = []

    for sentence, tuple_line in zip(entail_lines, tuple_lines):
        sentence = sentence.strip()
        tuple_line = tuple_line.strip()
        if not sentence:
            continue

        try:
            negated_sentence = negator.negate_sentence(sentence, prefer_contractions=False)
            updated_entail.append(f"{negated_sentence}\n")
            updated_tuple.append(f"{tuple_line}\n")
        except Exception as e:
            print(f"跳过文件 {txt_file}，原因：{e}")
            continue

    # 将更新后的内容写回文件
    with open(source_file, 'w', encoding='utf-8') as f_entail:
        f_entail.writelines(updated_entail)

    with open(tuple_file, 'w', encoding='utf-8') as f_tuple:
        f_tuple.writelines(updated_tuple)