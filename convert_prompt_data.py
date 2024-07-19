import json
import copy

def convert_ner_cluener_prompt(inputfile,outputfile):

    data = []

    entity_map = {'name':'人名',
                  'organization':'组织机构',
                  'scene':'景点',
                  'company':'企业',
                  'movie':'影视',
                  'book':'书籍',
                  'government':'政府',
                  'position':'职位',
                  'address':'地点',
                  'game':'游戏'}

    label = {
        "人名": '代表人名类实体，如：张三、刘德华、特朗普等',
        "组织机构": '代表组织机构类实体，如：中国足协、美国儿童基金会等',
        "景点": '代表景点类实体，如：故宫、西湖、敦煌莫高窟等',
        "企业": '代表企业类实体，如：京东、越南发展银行、杭州清风科技有限公司等',
        "影视": '代表影视类实体，如：《天下无贼》、英雄等',
        "书籍": '代表书籍类实体，如：红与黑、《活着》等',
        "政府": '代表政府类实体，如：印度外交部、发改委等',
        "职位": '代表职位类实体，如：老师、记者等',
        "地点": '代表地点类实体，如：北京、纽约、太平村等',
        "游戏": '代表游戏类实体，如：dota2、《使命召唤》等',
    }


    maxlen = 0
    max_source = 0
    max_target = 0

    with open(inputfile,'r',encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            text = line['text']

            query_json = {}
            for key in entity_map:
                new_key = '属于' + entity_map[key] + '类实体有?'
                query_json[new_key] = []

            output = copy.deepcopy(query_json)
            for key in line['label']:
                tmp = list(line['label'][key].keys())
                key_map = entity_map[key]
                for o in output:
                    if key_map in o:
                        output[o] = tmp

            # instruction = "这是命名实体识别任务，请根据给定原文“{}”，填写以下json：{}".format(text, query_json)
            instruction = "这是命名实体识别任务，其实体定义为：{}，请根据给定原文“{}”，填写以下json：{}".format(label, text, query_json)
            output =  "{}".format(output)

            data.append({"text": text, "instruction": instruction, "output": output, "task_type": "ner_cluener"})

            if len(instruction) + len(output) > maxlen:
                maxlen = len(instruction) + len(output)
            if len(instruction) > max_source:
                max_source = len(instruction)
            if len(output) > max_target:
                max_target = len(output)

    with open(outputfile, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False)+'\n')

    print('maxlen',maxlen)
    print('max_source',max_source)
    print('max_target',max_target)


def process_text(text,output,entity_map):
    text_list = list(text)
    for key in output:
        for w in output[key]:
            for item in output[key][w]:
                w_s = item[0]
                w_e = item[1]
                text_list[w_s] = entity_map[key][0] + text_list[w_s]
                text_list[w_e] = text_list[w_e] + entity_map[key][1]
    return ''.join(text_list)



def convert_ner_cluener_prompt4(inputfile,outputfile):

    entity_map = {'name': ['<name>','</name>'],
                  'organization': ['<organization>','</organization>'],
                  'scene': ['<scene>','</scene>'],
                  'company': ['<company>','</company>'],
                  'movie': ['<movie>','</movie>'],
                  'book': ['<book>','</book>'],
                  'government': ['<government>','</government>'],
                  'position': ['<position>','</position>'],
                  'address': ['<address>','</address>'],
                  'game': ['<game>','</game>']
                  }


    maxlen = 0
    max_source = 0
    max_target = 0
    data = []
    with open(inputfile, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            text = line['text']
            prefix = """这是一个命名实体识别任务，需要你参考<样例>，对给定的<文本>和<实体标签>信息，按<要求>，抽取<文本>包含的实体。

<文本>
{}

<实体标签>
<company>,</company>
<name>,</name>
<organization>,</organization>
<scene>,</scene>
<movie>,</movie>
<book>,</book>
<government>,</government>
<position>,</position>
<address>,</address>
<game>,</game>

<样例>
输入：浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，
输出：<company>浙商银行</company>企业信贷部<name>叶老桂</name>博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，

输入：生生不息CSOL生化狂潮让你填弹狂扫
输出：生生不息<game>CSOL</game>生化狂潮让你填弹狂扫

<要求>
1.提供的<实体标签>为<type>,</type>形式，表示type实体类型对应的开始与结束标签，需要你在<实体标签>限定的实体类型进行识别，不要识别之外的实体类型；
2.识别过程是判断<文本>中某个连续的片段若是实体，就用对应的实体起始标签进行标记；
3.输出形式参考<样例>中的输出；"""

            output = process_text(text, line['label'],entity_map)
            instruction = prefix.format(text)
            #print(instruction)
            #print(output)

            data.append({"text": text, "instruction": instruction, "output": output, "task_type": "ner_cluener"})

            if len(instruction) + len(output) > maxlen:
                maxlen = len(instruction) + len(output)
            if len(instruction) > max_source:
                max_source = len(instruction)
            if len(output) > max_target:
                max_target = len(output)

    with open(outputfile, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False)+'\n')

    print('maxlen',maxlen)
    print('max_source',max_source)
    print('max_target',max_target)



if __name__=="__main__":
    inputfile = r'E:\open_data\cluener_public\train.json'
    outputfile = r'E:\openlab\ChatGLM2-6B\data\ner\train2.json'
    convert_ner_cluener_prompt(inputfile,outputfile)


