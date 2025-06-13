import os
import requests
from bs4 import BeautifulSoup
import spacy
import random
# 加载spaCy英文模型
nlp = spacy.load("en_core_web_sm")

def get_synonyms_from_thesaurus(word):
    url = f"https://www.thesaurus.com/browse/{word}"
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/89.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.64',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/89.0',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/537.36 Edg/45.0.2254.21',
        'Mozilla/5.0 (Linux; Android 9; Pixel 3 XL) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; rv:34.0) Gecko/20100101 Firefox/34.0',
    ]
    headers = {
        'User-Agent': random.choice(user_agents),
    }
    try:
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code != 200:
            print(f"Failed to retrieve page for {word}")
            return []
        soup = BeautifulSoup(response.text, 'html.parser')
        # 查找包含"Strongest matches"的<p>标签
        strongest_matches_p = soup.find('p', string='Strongest matches')
        # 如果没有找到 "Strongest matches"，则查找 "Strongest match"
        if not strongest_matches_p:
            strongest_matches_p = soup.find('p', string='Strongest match')

        if strongest_matches_p:
            next_tag = strongest_matches_p.find_next()
            synonyms = [link.text.strip() for link in next_tag.find_all('a')]

            return synonyms
        else:
            print(f"Could not find 'Strongest matches' section for {word}")
            return []
    except requests.exceptions.RequestException as e:
        get_synonyms_from_thesaurus(word)

entail_only = 'entailonly_sim'
tuple = 'tuple_entailonly_sim'

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

        doc = nlp(sentence)
        modified_sentence = sentence
        first_noun = None
        for token in doc:
            if token.pos_ == "NOUN":
                first_noun = token.text
                synonyms = get_synonyms_from_thesaurus(first_noun)
                if synonyms:
                    print(f"替换名词: {first_noun} -> {synonyms[0]}")
                    modified_sentence = modified_sentence.replace(first_noun, synonyms[0], 1)  # 替换第一个找到的名词
                    break

        if first_noun and modified_sentence != sentence:
            print("替换后的句子:", modified_sentence)
            updated_entail.append(f"{modified_sentence}\n")
            updated_tuple.append(f"{tuple_line}\n")  # 保留tuple文件中的行
            sum += 1
        else:
            print(f"无法替换{txt_file}近义词，跳过{sentence}。")

    with open(source_file, 'w', encoding='utf-8') as f_entail:
        f_entail.writelines(updated_entail)

    with open(tuple_file, 'w', encoding='utf-8') as f_tuple:
        f_tuple.writelines(updated_tuple)

    if not updated_entail: 
        print(f"文件 {source_file} 和 {tuple_file} 为空，将被删除。")
        os.remove(source_file)
        os.remove(tuple_file)