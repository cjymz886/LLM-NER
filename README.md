# LLM-NER
LLM for NER

本次尝试利用大模型做下NER任务，看看大模型在信息抽取上能达到什么水准。由于笔者资源有限，本次实验主要在chatglm2-6b+ptuning，baichuan2-7b+lora方式进行微调和测试的，数据集选择CLUEbenchmark。<br/>

## 1、数据集转化
首先，数据集中实体类型名称为英文，统一转为中文，对应为：<br/>

```
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
```
接着，将数据集转化成指令类，本次尝试两次instruction方式，分别为instruction1、instruction2：<br/>

### instruction1
```
{
"text":"浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，",
"instruction":"这是命名实体识别任务，请根据给定原文“浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，”，填写以下json：{'属于人名类实体有?': [], '属于组织机构类实体有?': [], '属于景点类实体有?': [], '属于企业类实体有?': [], '属于影视类实体有?': [], '属于书籍类实体有?': [], '属于政府类实体有?': [], '属于职位类实体有?': [], '属于地点类实体有?': [], '属于游戏类实体有?': []}",
"output":"{'属于人名类实体有?': ['叶老桂'], '属于组织机构类实体有?': [], '属于景点类实体有?': [], '属于企业类实体有?': ['浙商银行'], '属于影视类实体有?': [], '属于书籍类实体有?': [], '属于政府类实体有?': [], '属于职位类实体有?': [], '属于地点类实体有?': [], '属于游戏类实体有?': []}",
"task_type":"ner_cluener"
}
```


### instruction2
```
{
"text":"浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，",
"instruction":"这是命名实体识别任务，其实体定义为：{'人名': '代表人名类实体，如：张三、刘德华、特朗普等', '组织机构': '代表组织机构类实体，如：中国足协、美国儿童基金会等', '景点': '代表景点类实体，如：故宫、西湖、敦煌莫高窟等', '企业': '代表企业类实体，如：京东、越南发展银行、杭州清风科技有限公司等', '影视': '代表影视类实体，如：《天下无贼》、英雄等', '书籍': '代表书籍类实体，如：红与黑、《活着》等', '政府': '代表政府类实体，如：印度外交部、发改委等', '职位': '代表职位类实体，如：老师、记者等', '地点': '代表地点类实体，如：北京、纽约、太平村等', '游戏': '代表游戏类实体，如：dota2、《使命召唤》等'}，请根据给定原文“浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，”，填写以下json：{'属于人名类实体有?': [], '属于组织机构类实体有?': [], '属于景点类实体有?': [], '属于企业类实体有?': [], '属于影视类实体有?': [], '属于书籍类实体有?': [], '属于政府类实体有?': [], '属于职位类实体有?': [], '属于地点类实体有?': [], '属于游戏类实体有?': []}",
"output":"{'属于人名类实体有?': ['叶老桂'], '属于组织机构类实体有?': [], '属于景点类实体有?': [], '属于企业类实体有?': ['浙商银行'], '属于影视类实体有?': [], '属于书籍类实体有?': [], '属于政府类实体有?': [], '属于职位类实体有?': [], '属于地点类实体有?': [], '属于游戏类实体有?': []}",
"task_type":"ner_cluener"
}
```

### instruction 3
```
{
"text":"浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，",
"instruction":"这是命名实体识别任务，需要分两步来识别：1）先识别出文本中存在的实体词；2）再判断实体词属于什么类别；实体类别集合为：['name', 'organization', 'scene', 'company', 'movie', 'book', 'government', 'position', 'address', 'game']，那么给定原文“浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，”，识别结果为：",
"output":"{'mention': ['浙商银行', '叶老桂'], 'entity': ['company', 'name']}",
"task_type":"ner_cluener"
}
```

### instruction 4
```
这是一个命名实体识别任务，需要你参考<样例>，对给定的<文本>和<实体标签>信息，按<要求>，抽取<文本>包含的实体。

<文本>
浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，

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
2.识别过程是判断<文本>中某个连续的片段是不是实体，若是，就用对应的实体起始标签进行标记；
3.输出形式参考<样例>中的输出
```

### 设计四种指令形式：<br/>

instruction1是一种问答的形式；<br/>

instruction2是在instruction1的基础上加了实体类别的举例与说明；<br/>

instruction3是采用COT形式，分两步来识别，先识别mention，再判断mention的类型；<br/>

instruction4是采样序列标记的方式，将实体的标签<type></type>嵌入文本中来标识实体，其更符合自回归生成的指令形式；<br/>


## 2、微调过程与结果
实验主要对比上述四种指令效果，其中_64，_128为p-tuning 中soft prompt设置超参数，​chatgml是采样P-tuning方式，baichuan2是采样lora方式。我们可以看出：我们可以看出：

![image](https://github.com/cjymz886/LLM-NER/blob/main/data/llm_ner.png)

### 结果分析 <br/>

（1）新增指令4在chatgml，baichuan两个基座下，都是取得最佳结果，新增的最佳结果F1值达到80.46，超过roberta-wwm-large-ext 的80.42；<br/>

（2）对比效果，#instruction 4 >#instruction 3>#instruction 1>#instruction 2，说明采用更符合自回归的生成方法（指令4）能更有利于信息抽取的任务，相对json的生成方式来说；<br/>

（3）指令3比指令1、指令2好，说明采用COT的生成方式也能带来提升，而指令2中虽然增加实体的举例与说明，但并没有带来明显的提升，可能增加这些信息反而是一种误导的噪声信息；<br/>

（4）基座中chatglm3比chatgm2，这其中是不是模型已经训练我们实验的数据集不得而知；另外，在p-tuning下，soft prompt参数越大，效果越好；<br/>

（5）在本实验中，显示lora下的结果比p-tuning好，但实验中基座没有保持一致，但直接说lora方法就比p-tuning，还不能断定。<br/>


## 3、执行步骤
1.执行：python convert_prompt_data.py，转化指令数据<br/>
2.在ptuning目录下，执行：bash train.sh， 训练<br/>
```
PRE_SEQ_LEN=64
LR=2e-2
NUM_GPUS=1
python main.py \
    --do_train \
    --train_file ../data/train.json \
    --preprocessing_num_workers 10 \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --output_dir output/ner/model1 \
    --overwrite_output_dir \
    --max_source_length 350 \
    --max_target_length 200 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
```
3.在ptuning目录下，执行：bash evaluate.sh， 预测<br/>
4.执行：python ner_eval.py 测评<br/>
本实验在单卡下跑的，若多卡参考ChatGLM2-6B；此外，测评显示是hard结果。<br/>




 参考
=
1. [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B/tree/main)






