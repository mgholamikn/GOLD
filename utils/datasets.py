import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, T5Tokenizer
from utils.ssl import no_ssl_verification
import numpy as np


# Define the TextEntailmentDataset class for RTE
class RTEDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        
        example = self.examples[index]
        premise = example['sentence1']
        hypothesis = example['sentence2']
        label = example['label']
        
        encoding = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'premise': premise,
            'hypothesis': hypothesis,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define the TextEntailmentDataset class for RTE
class RTEDataset2(Dataset):
    def __init__(self, examples, tokenizer, max_len):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        
        example = self.examples[index]
        premise = example['sentence1']
        hypothesis = example['sentence2']
        label = example['label']
        
        encoding = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            encoding['input_ids'].flatten(),
            torch.tensor(label, dtype=torch.long)
        }
    

# Define the TextEntailmentDataset class for QNLI
class QNLIDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_length_input, max_length_target,percent_data=100,task='qnli'):
        self.tokenizer = tokenizer
        self.data = dataset
        self.max_length_input = max_length_input
        self.max_length_target = max_length_target
        self.percent=percent_data
        self.task=task
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        example = self.data[index]

        input_text = self.task+' question: '+example['question']+' sentence: '+example['sentence']

        labels_map = ["entailment", "not_entailment"]
        if isinstance(example["label"],str):
            target_text=example["label"]     
        else:
            target_text=labels_map[example["label"]] 
            

        target_tokens = self.tokenizer.encode(target_text, max_length=self.max_length_target, padding='max_length', truncation=True) 

        input_ids = self.tokenizer.encode(input_text, padding='max_length', truncation=True, max_length=self.max_length_input)

        
        return {
            'input_ids': torch.tensor(input_ids),
            'label': torch.tensor(target_tokens),
            'label_text':target_text,
            'sentence_text':example['sentence'],
            'question_text':example['question'],
        }

# Define a custom dataset for LP problem description and formulation
class LPDataset(Dataset):
    def __init__(self, data,tokenizer,for_cgscore=False):
        self.data = data
        # with no_ssl_verification():
        self.tokenizer = tokenizer
        self.for_cgscore=for_cgscore

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        description = example['Problem']
        formulation = example['Formulation']

        input_text = 'Generate LP formulation of this problem: ' + description
        target_text = 'LP formulation: ' + formulation

        input_ids = self.tokenizer.encode(input_text, padding='max_length', truncation=True, max_length=512)
        target_ids = self.tokenizer.encode(target_text, padding='max_length', truncation=True, max_length=128)

        # # Tokenize the source and target texts
        # source_tokens = self.tokenizer.encode_plus(
        #     input_text,
        #     padding='max_length',
        #     truncation=True,
        #     max_length=512,
        #     return_tensors='pt'
        # )

        # target_tokens = self.tokenizer.encode_plus(
        #     target_text,
        #     padding='max_length',
        #     truncation=True,
        #     max_length=512,
        #     return_tensors='pt'
        # )


            
        if self.for_cgscore:
            return {torch.tensor(input_ids), torch.tensor(target_ids)}
        else:
        #     return {
        #         'input_ids': source_tokens['input_ids'].squeeze(),
        #         'attention_mask': source_tokens['attention_mask'].squeeze(),
        #         'decoder_input_ids': target_tokens['input_ids'].squeeze(),
        #         'decoder_attention_mask': target_tokens['attention_mask'].squeeze(),
        #         'label': target_tokens['input_ids'].squeeze(),
        #         'input_text':example['Problem'],
        #         'label_text':example['Formulation']
        #     }
            return {'input_ids':torch.tensor(input_ids), 'label':torch.tensor(target_ids),'input_text':example['Problem'],'label_text':example['Formulation']}




class ANLIDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_length_input, max_length_target,percent_data=100,label_map = ["entailment", "neutral","contradiction"],task='rte'):
        self.tokenizer = tokenizer
        self.data = dataset
        self.max_length_input = max_length_input
        self.max_length_target = max_length_target
        self.percent=percent_data
        self.label_map=label_map
        self.task=task
    
    def __len__(self):
        return int(len(self.data)/100*self.percent)
    
    def __getitem__(self, index):
        item = self.data[index]
     
        
        if isinstance(item["label"],str):
            target_text=item["label"]     
        else:
            target_text=self.label_map[item["label"]] 
        
        if self.task in ['mnli','anli']:
            sent1=' hypothesis:'
            sent2='premise:'
        else:        
            sent1=' sentence1:'
            sent2='sentence1:'

        if "premise" in item.keys():    
            input_text = self.task+sent1+ item["premise"] +sent2+ item["hypothesis"]
            sentence1 = item["premise"]
            sentence2 = item["hypothesis"]
        else:
            input_text = self.task+sent1+ item["sentence1"] +sent2+ item["sentence2"]
            sentence1 = item["sentence1"]
            sentence2 = item["sentence2"]


        input_ids = self.tokenizer.encode(input_text, padding='max_length', truncation=True, max_length=self.max_length_input)

        target_tokens = self.tokenizer.encode(target_text, max_length=self.max_length_target, padding='max_length', truncation=True)  
            
        return {
            'input_ids': torch.tensor(input_ids),
            'label': torch.tensor(target_tokens),
            'label_text':target_text,
            'sentence1_text': sentence1,
            'sentence2_text': sentence2
        }



class COLADataset(Dataset):
    def __init__(self, tokenizer, dataset, max_length_input, max_length_target,percent_data=100,label_map = ["unacceptable", "acceptable"]):
        self.tokenizer = tokenizer
        self.data = dataset
        self.max_length_input = max_length_input
        self.max_length_target = max_length_target
        self.percent=percent_data
        self.label_map=label_map
    
    def __len__(self):
        return int(len(self.data)/100*self.percent)
    
    def __getitem__(self, index):
        item = self.data[index]
     
        
        if isinstance(item["label"],str):
            target_text=item["label"]     
        else:
            target_text=self.label_map[item["label"]] 
        input_text = item["sentence"]
      

        input_ids = self.tokenizer.encode(input_text, padding='max_length', truncation=True, max_length=self.max_length_input)

        target_tokens = self.tokenizer.encode(target_text, max_length=self.max_length_target, padding='max_length', truncation=True)  
            
        return {
            'input_ids': torch.tensor(input_ids),
            'label': torch.tensor(target_tokens),
            'label_text':target_text
        }



class ANLIDataset2(Dataset):
    def __init__(self, dataset, tokenizer,  max_length_input, max_length_target):
        self.tokenizer = tokenizer
        self.data = dataset
        self.max_length_input = max_length_input
        self.max_length_target = max_length_target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
     
        target_text=item["label"]
        
        input_text = 'Sentence 1:' + item["sentence1"] +'Sentence 2:' + item["sentence2"] 

        input_ids = self.tokenizer.encode(input_text, padding='max_length', truncation=True, max_length=self.max_length_input)

        target_tokens = self.tokenizer.encode(target_text, max_length=self.max_length_target, padding='max_length', truncation=True)  
            
        return {
            'input_ids': torch.tensor(input_ids),
            'label': torch.tensor(target_tokens),
            'label_text':target_text,
            'sentence1_text': item["sentence1"],
            'sentence2_text': item["sentence2"],

        }



class squad_dataset(Dataset):
    def __init__(self,  tokenizer,dataset,  max_length_input, max_length_target,task='squad'):
        self.tokenizer = tokenizer
        self.data = dataset
        self.max_length_input = max_length_input
        self.max_length_target = max_length_target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        
        try:
            texts=item['answers']["text"]
            target_text='\n'
            for text in texts:
                target_text+=text
                target_text+='\n'
        except:
            target_text=item['answer']
       
        
        input_text = 'question: ' + item["question"] +' context: ' + item["context"] 

        input_tokens = self.tokenizer.encode_plus(input_text, padding='max_length', truncation=True, max_length=self.max_length_input)

        target_tokens = self.tokenizer.encode_plus(target_text, max_length=self.max_length_target, padding='max_length', truncation=True)  
        return {
        'input_ids': torch.tensor(input_tokens['input_ids']),
        'attention_mask': torch.tensor(input_tokens['attention_mask']),
        'label': torch.tensor(target_tokens['input_ids']),
        'decoder_attention_mask': torch.tensor(target_tokens['attention_mask']),
        'context': item["context"],
        'question': item["question"],
        'label_text': target_text,
        }

class svamp_dataset(Dataset):
    def __init__(self,  tokenizer,dataset,  max_length_input, max_length_target):
        self.tokenizer = tokenizer
        self.data = dataset
        self.max_length_input = max_length_input
        self.max_length_target = max_length_target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        
        if isinstance(item['Equation'],str):
            target_text=item['Equation']
        # else:
        #     target_text=str(int(item['Equation']))

       
        
        input_text = ' Body:' + item["Body"] +', Question:' + item["Question"] 

        input_tokens = self.tokenizer.encode_plus(input_text, padding='max_length', truncation=True, max_length=self.max_length_input)

        target_tokens = self.tokenizer.encode_plus(target_text, max_length=self.max_length_target, padding='max_length', truncation=True)  
        return {
        'input_ids': torch.tensor(input_tokens['input_ids']),
        'attention_mask': torch.tensor(input_tokens['attention_mask']),
        'label': torch.tensor(target_tokens['input_ids']),
        'decoder_attention_mask': torch.tensor(target_tokens['attention_mask']),
        'body': item["Body"],
        'question': item["Question"],
        # 'answer': str(item["Answer"]),
        'label_text': target_text,
        }    


class temporal_dataset(Dataset):
    def __init__(self,  tokenizer,dataset,  max_length_input, max_length_target):
        self.tokenizer = tokenizer
        self.data = dataset
        self.max_length_input = max_length_input
        self.max_length_target = max_length_target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        
        # try:
        #     texts=item['answers']["text"]
        #     target_text='\n'
        #     for text in texts:
        #         target_text+=text
        #         target_text+='\n'
        # except:
        target_text=item['Output']
       
        
        input_text = item["Input"] 

        input_tokens = self.tokenizer.encode_plus(input_text, padding='max_length', truncation=True, max_length=self.max_length_input)

        target_tokens = self.tokenizer.encode_plus(target_text, max_length=self.max_length_target, padding='max_length', truncation=True)  
        return {
        'input_ids': torch.tensor(input_tokens['input_ids']),
        'attention_mask': torch.tensor(input_tokens['attention_mask']),
        'label': torch.tensor(target_tokens['input_ids']),
        'decoder_attention_mask': torch.tensor(target_tokens['attention_mask']),
        'Input': item["Input"],
        'label_text': target_text,
        }


class mtc_dataset(Dataset):
    def __init__(self,  tokenizer,dataset,  max_length_input, max_length_target):
        self.tokenizer = tokenizer
        self.data = dataset
        self.max_length_input = max_length_input
        self.max_length_target = max_length_target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        
        # try:
        #     texts=item['answers']["text"]
        #     target_text='\n'
        #     for text in texts:
        #         target_text+=text
        #         target_text+='\n'
        # except:
        try:
            target_text=item['Note']
        except:
            target_text=item['Class']
       
        
        input_text = item["Dialogue"] 

        input_tokens = self.tokenizer.encode_plus(input_text, padding='max_length', truncation=True, max_length=self.max_length_input)

        target_tokens = self.tokenizer.encode_plus(target_text, max_length=self.max_length_target, padding='max_length', truncation=True)  
        return {
        'input_ids': torch.tensor(input_tokens['input_ids']),
        'attention_mask': torch.tensor(input_tokens['attention_mask']),
        'label': torch.tensor(target_tokens['input_ids']),
        'decoder_attention_mask': torch.tensor(target_tokens['attention_mask']),
        'Dialogue': item["Dialogue"],
        'label_text': target_text,
        }


class nl4opt_dataset(Dataset):
    def __init__(self,  tokenizer,dataset,  max_length_input, max_length_target):
        self.tokenizer = tokenizer
        self.data = dataset
        self.max_length_input = max_length_input
        self.max_length_target = max_length_target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        
        # try:
        #     texts=item['answers']["text"]
        #     target_text='\n'
        #     for text in texts:
        #         target_text+=text
        #         target_text+='\n'
        # except:

        target_text=item['formulation']

       
        
        input_text = item["problem"] 

        input_tokens = self.tokenizer.encode_plus(input_text, padding='max_length', truncation=True, max_length=self.max_length_input)

        target_tokens = self.tokenizer.encode_plus(target_text, max_length=self.max_length_target, padding='max_length', truncation=True)  
        return {
        'input_ids': torch.tensor(input_tokens['input_ids']),
        'attention_mask': torch.tensor(input_tokens['attention_mask']),
        'label': torch.tensor(target_tokens['input_ids']),
        'decoder_attention_mask': torch.tensor(target_tokens['attention_mask']),
        'problem': item["problem"],
        'formulation': target_text,
        'label_text': target_text,
        }

def prepare_data(dataset_name='rte',tokenizer=None,max_length_input=128,max_length_target=128,percent_data=100):
    if dataset_name in ['rte','wnli']:
        with no_ssl_verification():
            # Load the RTE dataset
            dataset = load_dataset('glue', dataset_name)
        if dataset_name=='rte':
            label_map=['entailment','not_entailment']
        elif dataset_name=='wnli':
            label_map=['not_entailment','entailment']
        train_dataset = ANLIDataset(tokenizer, dataset['train'],max_length_input=max_length_input,max_length_target=max_length_target,percent_data=percent_data,label_map=label_map,task='rte')
        test_dataset = ANLIDataset(tokenizer, dataset['validation'],max_length_input=max_length_input,max_length_target=max_length_target,label_map=label_map,task='rte')
        
    
    elif dataset_name=='qnli':
        # Load the QNLI dataset
        dataset = load_dataset('glue', 'qnli')
        train_dataset = QNLIDataset(tokenizer, dataset['train'],max_length_input=max_length_input,max_length_target=max_length_target,percent_data=percent_data,task=dataset_name)
        test_dataset = QNLIDataset(tokenizer, dataset['validation'],max_length_input=max_length_input,max_length_target=max_length_target,task=dataset_name)
        
    elif dataset_name=='anli':
        with no_ssl_verification():
            # tokenizer = T5Tokenizer.from_pretrained('t5-base')
            # Load the ANLI dataset from Hugging Face
            train_dataset = load_dataset("anli",split='train_r1')
            test_dataset = load_dataset("anli",split='test_r1')
            # Create the ANLIDataset instances for training and test sets
            train_dataset = ANLIDataset(tokenizer, train_dataset,max_length_input=max_length_input,max_length_target=max_length_target,percent_data=percent_data,task='mnli')
            test_dataset = ANLIDataset(tokenizer, test_dataset,max_length_input=max_length_input,max_length_target=max_length_target,task='mnli')
    elif dataset_name=='mnli':
        with no_ssl_verification():
            # Load the MNLI dataset
            dataset = load_dataset('glue', 'mnli')
        test_dataset = ConcatDataset([dataset['validation_mismatched'], dataset['validation_matched']])
        train_dataset = ANLIDataset(tokenizer, dataset['train'],max_length_input=max_length_input,max_length_target=max_length_target,percent_data=percent_data,task=dataset_name)
        test_dataset = ANLIDataset(tokenizer, test_dataset,max_length_input=max_length_input,max_length_target=max_length_target,task=dataset_name)
        
    elif dataset_name=='snli':
        with no_ssl_verification():
            train_dataset = load_dataset('snli',split='train')
            test_dataset = load_dataset('snli',split='test')

            train_dataset = ANLIDataset(tokenizer, train_dataset,max_length_input=max_length_input,max_length_target=max_length_target)
            test_dataset = ANLIDataset(tokenizer, test_dataset,max_length_input=max_length_input,max_length_target=max_length_target)
    elif dataset_name=='squad':
        with no_ssl_verification():
            dataset = load_dataset('squad')
            train_dataset = dataset['train']
            test_dataset = dataset['validation']

            train_dataset = squad_dataset(tokenizer, train_dataset,max_length_input=max_length_input,max_length_target=max_length_target,task=dataset_name)
            test_dataset = squad_dataset(tokenizer, test_dataset,max_length_input=max_length_input,max_length_target=max_length_target,task=dataset_name)
    
    elif dataset_name=='adversarialQA':
       with no_ssl_verification():
            dataset = load_dataset('adversarial_qa', 'adversarialQA')
            train_dataset = dataset['train']
            test_dataset = dataset['validation']

            train_dataset = squad_dataset(tokenizer, train_dataset,max_length_input=max_length_input,max_length_target=max_length_target)
            test_dataset = squad_dataset(tokenizer, test_dataset,max_length_input=max_length_input,max_length_target=max_length_target)

    elif dataset_name=='svamp':
       with no_ssl_verification():
            dataset = load_dataset('data/svamp')
            train_dataset = dataset['train']
            test_dataset = dataset['test']

            train_dataset = svamp_dataset(tokenizer, train_dataset,max_length_input=max_length_input,max_length_target=max_length_target)
            test_dataset = svamp_dataset(tokenizer, test_dataset,max_length_input=max_length_input,max_length_target=max_length_target)


    elif dataset_name=='cola':
        with no_ssl_verification():
            # Load the MNLI dataset
            dataset = load_dataset('glue', 'cola')
        train_dataset = COLADataset(tokenizer, dataset['train'],max_length_input=max_length_input,max_length_target=max_length_target,percent_data=percent_data)
        test_dataset = COLADataset(tokenizer, dataset['validation'],max_length_input=max_length_input,max_length_target=max_length_target)

    elif dataset_name=='mrpc':
        with no_ssl_verification():
            # Load the MNLI dataset
            dataset = load_dataset('glue', 'mrpc')
        label_map=['not_equivalent','equivalent']
        train_dataset = ANLIDataset(tokenizer, dataset['train'],max_length_input=max_length_input,max_length_target=max_length_target,percent_data=percent_data,label_map=label_map,task=dataset_name)
        test_dataset = ANLIDataset(tokenizer, dataset['validation'],max_length_input=max_length_input,max_length_target=max_length_target,label_map=label_map,task=dataset_name)

    elif dataset_name=='temporal':
       with no_ssl_verification():
            dataset = load_dataset('data/temporal')
            train_dataset = dataset['train']
            test_dataset = dataset['test']

            train_dataset = temporal_dataset(tokenizer, train_dataset,max_length_input=max_length_input,max_length_target=max_length_target)
            test_dataset = temporal_dataset(tokenizer, test_dataset,max_length_input=max_length_input,max_length_target=max_length_target)

    elif dataset_name=='mtc':
       with no_ssl_verification():
            dataset = load_dataset('data/mtc')
            train_dataset = dataset['train']
            test_dataset = dataset['test']

            train_dataset = mtc_dataset(tokenizer, train_dataset,max_length_input=max_length_input,max_length_target=max_length_target)
            test_dataset = mtc_dataset(tokenizer, test_dataset,max_length_input=max_length_input,max_length_target=max_length_target)

    elif dataset_name=='mtc2':
       with no_ssl_verification():
            dataset = load_dataset('data/mtc2')
            train_dataset = dataset['train']
            test_dataset = dataset['test']

            train_dataset = mtc_dataset(tokenizer, train_dataset,max_length_input=max_length_input,max_length_target=max_length_target)
            test_dataset = mtc_dataset(tokenizer, test_dataset,max_length_input=max_length_input,max_length_target=max_length_target)

    elif dataset_name=='nl4opt':
       with no_ssl_verification():
            dataset = load_dataset('data/nl4opt')
            train_dataset = dataset['train']
            test_dataset = dataset['test']

            train_dataset = nl4opt_dataset(tokenizer, train_dataset,max_length_input=max_length_input,max_length_target=max_length_target)
            test_dataset = nl4opt_dataset(tokenizer, test_dataset,max_length_input=max_length_input,max_length_target=max_length_target)

    return train_dataset,test_dataset 
