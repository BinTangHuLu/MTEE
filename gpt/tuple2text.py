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
def tuple2text(tuples):
    # 通过 GPT-3.5 API 进行请求
    prompt = (
            "Task: Please generate a simple sentence for each given tuple. The types of tuples are as follows:\n"
            "Entity: A noun or noun phrase representing a concrete or abstract object.\n"
            "Attribute: A property or characteristic describing an entity. \n"
            "Relation: A semantic connection between two entities.\n"
            
            "Do not introduce any extra semantics. Only output the sentences.\n"
            
            "Here are some examples:\n"
            "Input1:\n"
            "entity,motorcycle\n"
            "entity,doors\n"
            "attribute,motorcycle,blue\n"
            "attribute,doors,paint chipped\n"
            "attribute,motorcycle,parked\n"
            "relation,motorcycle,parked by,doors\n"
            "Output1:\n"
            "There is a motorcycle.\n"
            "There are some doors.\n"
            "The motorcycle is blue.\n"
            "The doors are paint chipped.\n"
            "The motorcycle is parked.\n"
            "The motorcycle is parked by doors.\n"
            
            "Input2:\n"
            "entity,stage\n"
            "attribute,children,7 year old\n"
            "relation,children,on,stage\n"
            "Output2:\n"
            "There is a stage.\n"
            "The children are 7 year old.\n"
            "The children are on stage.\n"
            
            "Input3:\n"
            "entity,men\n"
            "entity,women\n"
            "attribute,men,four\n"
            "attribute,women,two\n"
            "Output3:\n"
            "There are men.\n"
            "There are women.\n"
            "There are four men.\n"
            "There are two women.\n"

            f"Tuples:{tuples}\n"
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
            tuple2text(tuples)

img_text_dir = 'data/img_text'
dataset_name = os.path.dirname(img_text_dir)#数据集的路径 如：dataset/VOC2007
dataset_img=os.path.basename(img_text_dir)#数据集下的图片文件夹名称 如：JPEGImages
text_tuple_dir = dataset_name + '/text_tuple'
tuple2text_dir = dataset_name + '/tuple2text'
isExists = os.path.exists(tuple2text_dir)
# 如果不存在则创建，否则清楚内容
if not isExists:
    os.mkdir(tuple2text_dir)
# else:
#     clear_folder(text_tuple_dir)


txt_files = [f for f in os.listdir(img_text_dir) if f.lower().endswith('.txt')]

# 一张图片可以有多个描述，所以先找txt
for txt_file in txt_files:
    txt_path = os.path.join(img_text_dir, txt_file)
    text_tuple_path = os.path.join(text_tuple_dir, txt_file)
    print('-'*100)
    print('原始tuple路径：', text_tuple_path)  # 原tuple路径
    with open(text_tuple_path, 'r', encoding='utf-8') as f:
        tuples = f.read().strip()
        print(tuples)
    tuple2text_path = os.path.join(tuple2text_dir, txt_file)
    print('衍生texts路径：', tuple2text_path)  # 衍生texts路径
    if os.path.exists(tuple2text_path):
        print(f"已存在衍生句子结果，跳过文件: {txt_file}")
        continue
    texts = tuple2text(tuples)
    print(texts)
    with open(tuple2text_path, 'w', encoding='utf-8') as f:
        f.write(texts)
        print('衍生texts结果已保存')
