import os

import nltk

entailonly_dir = 'data/entailonly_sim'
tuple_dir = 'data/tuple_entailonly_sim'
hypothesis_dir = 'data/hypothesis'

cause_error_dir = 'data/entailonly_sim_error'
isExists = os.path.exists(cause_error_dir)
# 如果不存在则创建
if not isExists:
    os.mkdir(cause_error_dir)

txt_files = [f for f in os.listdir(entailonly_dir) if f.lower().endswith('.txt')]

for txt_file in txt_files:
    text_path = os.path.join(entailonly_dir, txt_file)
    tuple_path = os.path.join(tuple_dir, txt_file)
    hypothesis_path = os.path.join(hypothesis_dir, txt_file)
    print('-'*80)
    print(text_path)
    with open(text_path, 'r', encoding='utf-8') as f:
        all_text_result = f.read().strip()
    error_cause_tuples = []
    if 'neutral' in all_text_result or 'contradiction' in all_text_result:
        with open(text_path, 'r', encoding='utf-8') as f1, open(tuple_path, 'r', encoding='utf-8') as f2:
            for line1, line2 in zip(f1, f2):  # 使用zip同时迭代两个文件
                if line1.strip().split('||')[1] in ['contradiction', 'neutral']:

                    error_cause_tuples.append((line2.strip()))
        print(error_cause_tuples)
        error_words = []
        error_entity = []
        error_attribute = []
        error_relation = []
        for tuple in error_cause_tuples:
            if tuple.strip().split(',')[0] == 'entity':
                # 提取实体
                entity = tuple.strip().split(',')[1].strip()
                # 将整个实体添加到error_entity
                error_entity.append(entity)
                words = entity.split()
                error_words.extend(words)

        for tuple in error_cause_tuples:
            if tuple.strip().split(',')[0] == 'attribute':
                # 实体和属性
                entity = tuple.strip().split(',')[1].strip()
                attribute = tuple.strip().split(',')[2].strip()
                if entity in error_entity:
                    words = attribute.split()
                    error_words.extend(words)
                else:
                    words = attribute.split()
                    error_words.extend(words)
            if tuple.strip().split(',')[0] == 'relation':
                # 实体和关系
                # entity1 是第一个实体
                entity1 = tuple.strip().split(',')[1].strip()
                # entity2 是最后一个部分
                entity2 = tuple.strip().split(',')[-1].strip()
                # relation 是中间的部分
                relation = ' '.join(tuple.strip().split(',')[2:-1]).strip()
                if entity1 in error_entity and entity2 in error_entity:
                    words = relation.split()
                    error_words.extend(words)
                if entity1 in error_entity and entity2 not in error_entity:
                    words = relation.split()
                    error_words.extend(words)
                if entity1 not in error_entity and entity2 in error_entity:
                    words = relation.split()
                    error_words.extend(words)
                if entity1 not in error_entity and entity2 not in error_entity:
                    words = relation.split()
                    error_words.extend(words)
        # print(error_words)
        hypothesis_path = os.path.join(hypothesis_dir, txt_file)
        with open(hypothesis_path, 'r', encoding='utf-8') as f:
            hypothesis = f.read().strip()
        hypothesis_words = nltk.word_tokenize(hypothesis.lower())
        # 创建一个新列表来存储处理后的单词
        processed_words = []
        for i, word in enumerate(hypothesis_words):
            if i > 0 and word == "'s":
                processed_words[-1] = processed_words[-1] + word
            else:
                processed_words.append(word)

        common_words = [word for word in processed_words if word.lower() in [e.lower() for e in error_words]]
        print(common_words)

        if len(common_words) > 0:
            error_cause_path = os.path.join(cause_error_dir, txt_file)
            with open(error_cause_path, 'w', encoding='utf-8') as f:
                f.write(" ".join(common_words))
