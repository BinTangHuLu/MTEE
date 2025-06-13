import os
import shutil

import yaml
import torch
from torchvision import transforms
from models.model_ve import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
import re
from PIL import Image
from dataset import create_dataset, create_sampler, create_loader

config_path = './configs/VE.yaml'
text_encoder = './cfg_bert_base'

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained(text_encoder)
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
model = ALBEF(text_encoder=text_encoder, tokenizer=tokenizer, config=config)

checkpoint_path = 'output/VE0.7790/checkpoint_best.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
state_dict = checkpoint['model']

# Reshape positional embedding to accommodate for image resolution change
pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
if config['distill']:
    m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
    state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

for key in list(state_dict.keys()):
    if 'bert' in key:
        new_key = key.replace('bert.', '')
        state_dict[new_key] = state_dict[key]
        del state_dict[key]

msg = model.load_state_dict(state_dict, strict=False)
print('Loaded checkpoint from %s' % checkpoint_path)
print(msg)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
test_transform = transforms.Compose([
    transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
    ])
def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption

def clear_folder(path):
    # 判断路径是否存在
    if os.path.exists(path):
        # 遍历文件夹中的所有文件和子文件夹
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)

            # 判断是否为文件，如果是则删除
            if os.path.isfile(file_path):
                os.remove(file_path)

            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        print(f"指定路径 '{path}' 不存在！")

tuple_text_dir ='data/entailonly_sim'
img_path = 'data/img'

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
    # 处理image
    image = Image.open(img).convert('RGB')
    image = test_transform(image)
    image = image.unsqueeze(0)
    print("测试img路径:" + img)

    with open(tuple_text_path, 'r', encoding='utf-8') as f:
        txt = ''
        for line in f:
            text = line.strip()
            # 跳过空行
            if not text:
                continue
            print(text)
            text = pre_caption(text, 30)

            labels = {'entailment': 2, 'neutral': 1, 'contradiction': 0}
            result = {2: 'entailment', 1: 'neutral', 0: 'contradiction'}
            # 处理target 如tensor([0])
            target = labels["entailment"]
            target = torch.tensor([target])

            image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
            text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)

            prediction = model(image, text_inputs, targets=target, train=False)
            _, pred_class = prediction.max(1)
            print('result: ', result[pred_class.item()])


            txt += f"{text} ||{result[pred_class.item()]}\n"
    with open(tuple_text_path, 'w', encoding='utf-8') as f:
        f.write(txt)