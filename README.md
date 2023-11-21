# LLM-NER
LLM for NER

本次尝试利用大模型做下NER任务，看看大模型在信息抽取上能达到什么水准。由于笔者资源有限，本次实验是在chatglm2-6b+ptuning方式进行微调和测试的，数据集选择CLUEbenchmark。<br/>

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
{ "text":"浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，", "instruction":"这是命名实体识别任务，请根据给定原文“浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，”，填写以下json：{'属于人名类实体有?': [], '属于组织机构类实体有?': [], '属于景点类实体有?': [], '属于企业类实体有?': [], '属于影视类实体有?': [], '属于书籍类实体有?': [], '属于政府类实体有?': [], '属于职位类实体有?': [], '属于地点类实体有?': [], '属于游戏类实体有?': []}", "output":"{'属于人名类实体有?': ['叶老桂'], '属于组织机构类实体有?': [], '属于景点类实体有?': [], '属于企业类实体有?': ['浙商银行'], '属于影视类实体有?': [], '属于书籍类实体有?': [], '属于政府类实体有?': [], '属于职位类实体有?': [], '属于地点类实体有?': [], '属于游戏类实体有?': []}", "task_type":"ner_cluener" }
```

### instruction2
```
{ "text":"浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，", "instruction":"这是命名实体识别任务，其实体定义为：{'人名': '代表人名类实体，如：张三、刘德华、特朗普等', '组织机构': '代表组织机构类实体，如：中国足协、美国儿童基金会等', '景点': '代表景点类实体，如：故宫、西湖、敦煌莫高窟等', '企业': '代表企业类实体，如：京东、越南发展银行、杭州清风科技有限公司等', '影视': '代表影视类实体，如：《天下无贼》、英雄等', '书籍': '代表书籍类实体，如：红与黑、《活着》等', '政府': '代表政府类实体，如：印度外交部、发改委等', '职位': '代表职位类实体，如：老师、记者等', '地点': '代表地点类实体，如：北京、纽约、太平村等', '游戏': '代表游戏类实体，如：dota2、《使命召唤》等'}，请根据给定原文“浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，”，填写以下json：{'属于人名类实体有?': [], '属于组织机构类实体有?': [], '属于景点类实体有?': [], '属于企业类实体有?': [], '属于影视类实体有?': [], '属于书籍类实体有?': [], '属于政府类实体有?': [], '属于职位类实体有?': [], '属于地点类实体有?': [], '属于游戏类实体有?': []}", "output":"{'属于人名类实体有?': ['叶老桂'], '属于组织机构类实体有?': [], '属于景点类实体有?': [], '属于企业类实体有?': ['浙商银行'], '属于影视类实体有?': [], '属于书籍类实体有?': [], '属于政府类实体有?': [], '属于职位类实体有?': [], '属于地点类实体有?': [], '属于游戏类实体有?': []}", "task_type":"ner_cluener" }
```

## 2、微调过程与结果

两种指令差别在于对实体类型的解释与说明，instruction2对10类实体都做了举例说明，以期望这类先验信息对任务有帮助。两种指令下微调训练时主要参数为：<br/>
| 参数 | instruction1 | instruction2|
| ------| ------| ------|
|PRE_SEQ_LEN|64/128|64|
|LR|2e-2|2e-2|
|max_source_length|350|512|
|max_target_length|200|200|
|max_steps|3000|3000|

两种指令训练的结果在验证集上结果为：<br/>

|  | F1 | Precision|Recall |
| ------| ------| ------| ------|
|instruction1_64|76.6|77.77|75.45|
|instruction2_64|75.95|77.87|74.11|
|instruction1_128|78.69|80.43|77.03|

最好的结果(instruction1)对比之前抽取模型：<br/>
|model|F1|
|---|---|
|bilistm+crf|	70.0|
|roberta-wwm-large-ext|	80.42|
|LLM(chatglm2-6b+ptuning)|	78.69|

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






