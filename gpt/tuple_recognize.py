import os
import time

import openai
from openai import OpenAI

# Set the proxy URL and port
proxy_url = ''
proxy_port = ''
# Set the http_proxy and https_proxy environment variables
os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'

client = OpenAI(
    base_url="",
    api_key="api-key"
)


# 定义任务
def identify_tuples(sentence):
    # 通过 GPT-3.5 API 进行请求
    prompt = (
            "Task: Please extract three types of tuple from the given sentence. The types of tuples are as follows: \n"
            "Entity: A noun or noun phrase representing a concrete or abstract object.\n"
            "Attribute: A property or characteristic describing an entity. \n"
            "Relation: A semantic connection between two entities.\n"
    
            "Do not generate same tuples again. Only ouput the tuples."
            
            "Here are some examples:\n"
            "Input1: A blue motorcycle parked by paint chipped doors.\n"
            "Output1: entity,motorcycle\n"
            "entity,doors\n"
            "attribute,motorcycle,blue\n"
            "attribute,doors,paint chipped\n"
            "attribute,motorcycle,parked\n"
            "relation,motorcycle,parked by,doors\n"
    
            "Input2: There are 7 year old children on stage.\n"
            "Output2: entity,children\n"
            "entity,stage\n"
            "attribute,children,7 year old\n"
            "relation,children,on,stage\n"
    
            "Input3: A beach scene with a man doing yoga.\n"
            "Output3: entity,scene\n"
            "entity,beach\n"
            "entity,man\n"
            "attribute,man,doing yoga\n"
            "relation,scene,with,man\n"

            "Input4: The man is wearing yoga pants.\n"
            "Output4: entity,man\n"
            "entity,pants\n"
            "attribute,pants,yoga\n"
            "relation,man,is wearing,pants\n"
            
            "Input5: The man is in black.\n"
            "Output5: entity,man\n"
            "attribute,man,in black\n"

            f"Sentence:{sentence}"
    )
    # 使用 ChatCompletion API
    max_retries = 6  # 最大重试次数
    retries = 0
    while retries < max_retries:
        try:
            # 使用 ChatCompletion API，并设置超时
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # 或使用 gpt-4 等其他模型
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
                timeout=60  # 设置超时为60秒
            )
            # 解析并返回结果
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"发生未知错误: {e}")
            time.sleep(10)
            identify_tuples(sentence)


img_text_dir = 'data/img_text'
dataset_name = os.path.dirname(img_text_dir)#数据集的路径 如：dataset/VOC2007
dataset_img=os.path.basename(img_text_dir)#数据集下的图片文件夹名称 如：JPEGImages
text_tuple_dir = dataset_name + '/text_tuple'
isExists = os.path.exists(text_tuple_dir)
# 如果不存在则创建，否则清楚内容
if not isExists:
    os.mkdir(text_tuple_dir)
# else:
#     clear_folder(text_tuple_dir)


txt_files = [f for f in os.listdir(img_text_dir) if f.lower().endswith('.txt')]

# 一张图片可以有多个描述，所以先找txt
for txt_file in txt_files:
    txt_path = os.path.join(img_text_dir, txt_file)
    print('-'*100)
    print('原始txt路径：', txt_path)  # 原txt路径
    text_tuple_path = os.path.join(text_tuple_dir, txt_file)
    if os.path.exists(text_tuple_path):
        print(f"已存在识别结果，跳过文件: {txt_file}")
        continue
    print('tuple结果txt路径：', text_tuple_path)  # tuple结果txt路径
    with open(txt_path, 'r', encoding='utf-8') as f:
        sentence = f.read().strip()
        print(sentence)
    # 获取元组
    tuples = identify_tuples(sentence)
    print(tuples)
    with open(text_tuple_path, 'w', encoding='utf-8') as f:
        f.write(tuples)
        print('tuple识别结果已保存')