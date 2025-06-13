import os
import pickle
import string

import torch

from esim.data import NLIDataset, Preprocessor
from esim.model import ESIM
pretrained_file = 'data/checkpoints/SNLI/best.pth.tar'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(pretrained_file, map_location=torch.device('cpu'), weights_only=True)
# Retrieving model parameters from checkpoint.
vocab_size = checkpoint["model"]["_word_embedding.weight"].size(0)
embedding_dim = checkpoint["model"]['_word_embedding.weight'].size(1)
hidden_size = checkpoint["model"]["_projection.0.weight"].size(0)
num_classes = checkpoint["model"]["_classification.4.weight"].size(0)
model = ESIM(vocab_size,
             embedding_dim,
             hidden_size,
             num_classes=num_classes,
             device=device).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()
preprocessor = Preprocessor(lowercase=True,
                            ignore_punctuation=True,
                            num_words=None,
                            stopwords=[],
                            labeldict={"entailment": 0,
                                       "neutral": 1,
                                       "contradiction": 2},
                            bos="_BOS_",
                            eos="_EOS_")
with open(os.path.join("data/preprocessed/SNLI/worddict.pkl"), "rb") as f:
    worddict = pickle.load(f)
preprocessor.worddict = worddict

def words_to_indices(preprocessor, sentence):
    indices = []
    parentheses_table = str.maketrans({"(": None, ")": None})
    punct_table = str.maketrans({key: " "
                                 for key in string.punctuation})

    sentence = sentence.translate(parentheses_table)
    if preprocessor.lowercase:
        sentence = sentence.lower()
    if preprocessor.ignore_punctuation:
        sentence = sentence.translate(punct_table)
    words = [w for w in sentence.rstrip().split() if w not in preprocessor.stopwords]

    if preprocessor.bos:
        indices.append(preprocessor.worddict["_BOS_"])
    for word in words:
        if word in preprocessor.worddict:
            index = preprocessor.worddict[word]
        else:
            index = preprocessor.worddict["_OOV_"]
        indices.append(index)
    if preprocessor.eos:
        indices.append(preprocessor.worddict["_EOS_"])
    return indices
def inference(p, h):
    premise_indices = words_to_indices(preprocessor, p)
    hypothesis_indices = words_to_indices(preprocessor, h)
    premise_tensor = torch.tensor(premise_indices).unsqueeze(0).to(device)
    hypothesis_tensor = torch.tensor(hypothesis_indices).unsqueeze(0).to(device)
    # 计算句子的长度
    premise_length = torch.tensor([len(premise_indices)]).to(device)
    hypothesis_length = torch.tensor([len(hypothesis_indices)]).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        # 进行前向推理
        _, probs = model(premise_tensor, premise_length, hypothesis_tensor, hypothesis_length)
        _, out_classes = probs.max(dim=1)
    # 显示预测类别
    if out_classes == 0:
        return 'entailment'
    elif out_classes == 2:
        return 'contradiction'
    else:
        return 'neutral'

tuple_text_dir ='data/entailonly_sim'
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
            inference_result = inference(premise_text, text)
            print('result：', inference_result)
            txt += f"{text} ||{inference_result}\n"
    with open(tuple_text_path, 'w', encoding='utf-8') as f:
        f.write(txt)