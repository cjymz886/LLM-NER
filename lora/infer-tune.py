import os
import math
import pathlib
from typing import Optional, Dict
from dataclasses import dataclass, field
import json
import time

import torch
from torch.utils.data import Dataset
import transformers
from transformers.training_args import TrainingArguments

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
os.environ["WANDB_DISABLED"] = "true"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=r"E:\pretraing_models\torch\baichuan2-7B-Chat")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    max_source_length: int = field(
        default=1000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_target_length: int = field(
        default=200,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    use_lora: bool = field(default=True)
    model_max_length: int = field(
        default=1201,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    



class MySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        max_source_length,
        max_target_length,
        max_seq_length
    ):
        super(MySupervisedDataset, self).__init__()
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_seq_length
        self.ignore_index = -100

    def load_data(self,data_path):
        D = []
        with open(data_path,'r',encoding='utf-8') as f:
            for line in f :
                line = json.loads(line)
                D.append(line)
        return D

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):

        # prompt, answer = example['instruction'], example['output']
        prompt = example['instruction']
        # cause = example['cause']
        # effect = example['effect']
        # event_type = example['event_type']
        # text = example['text']
        labels = example['output']
        model_inputs = self.tokenizer([prompt], max_length=self.max_source_length, truncation=True,add_special_tokens=True)
        return {'input_ids':model_inputs['input_ids'],'labels':labels}


    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])









def predict():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    model.half().cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    if training_args.use_lora:
        from peft import PeftModel
        from peft import LoraConfig, TaskType, get_peft_model
        peft_path = r"E:\openlab\Baichuan2\fine-tune\output\m_ctx"
        model = PeftModel.from_pretrained(model, peft_path)

    dataset = MySupervisedDataset(
        data_args.data_path, tokenizer, data_args.max_source_length,data_args.max_target_length,training_args.model_max_length
    )
    f = open(r'E:\openlab\Baichuan2\fine-tune\data\pkumod-ccks_query_list_test_preidct.txt','w',encoding='utf-8')
    model.eval()
    with torch.autocast("cuda"):
        k = 0
        for line in dataset:
            labels = line['labels']
            input_ids = torch.LongTensor(line['input_ids']).to('cuda')
            
            # def allowed_fn(b, ts):
            #     return input_ids

            # force_words_ids = line['input_ids']

            out = model.generate(
                input_ids=input_ids,
                # max_length=1024,
                do_sample=True,
                max_new_tokens = 200,
                # num_beams=2,
                temperature = 0.3,
                top_k = 5,
                top_p= 0.85,
                eos_token_id=tokenizer.eos_token_id,
                # force_words_ids=force_words_ids,
                # prefix_allowed_tokens_fn = allowed_fn,
            )

            out = out[:,input_ids.size()[-1]:]
            out_text = tokenizer.decode(out[0],skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            out_text = out_text.split('</s>')[0]
            out_text = out_text.split('\n\n')[0]
            # print('__________________')
            # print(out_text)
            f.write(json.dumps({'labels':labels,'output':out_text},ensure_ascii=False)+'\n')
            k+=1
            if k%10==0:
                print('having predict %dtd data'%k)
                # break
    f.close()



if __name__ == "__main__":
    predict()

