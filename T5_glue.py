
### scenario 0:
## python T5_glue.py --scenario0 --num_batches 375 --data_dir generated_data/rte_ours/ --epochs 1 --generate_data  --dataset rte --fb val_ac 

### scenario 1:
## CUDA_VISIBLE_DEVICES=0 nohup python T5_glue.py --scenario1 --num_batches 2125 --data_dir generated_data/anli_17k_syntheticdata_woFB/ 


### scenario 4:
## python T5_glue.py --scenario4 --data_dir generated_data/qnli/ --num_batches 375 --epochs 50  --dataset qnli  --lr 1e-5 --loss sce --alpha 0.01 

### scenario 6:
## python T5_glue.py --scenario6 --dataset qnli 



import os
os.environ['CURL_CA_BUNDLE'] = ''
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AutoTokenizer
from transformers import AdamW
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from utils.ssl import *
from utils.datasets import *
from utils.common import *
from tqdm import tqdm
import json
from utils.llm import *
import argparse
from tqdm import tqdm  
from utils.batching import *
import glob
from random import shuffle
import transformers
from CG_score.cg_function import calc_cg_score
from utils.common import energy_score


torch.manual_seed(44)

parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--dataset',type=str,default='anli')
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--lr',type=float,default=5e-5)
parser.add_argument('--bs',type=int,default=8)
parser.add_argument('--pd',type=float,default=100,help='percent of real data to be used when in the case of train_real or scenario 2')
parser.add_argument('--scenario0',action='store_true', help='generate synthetic data with FB')
parser.add_argument('--scenario4',action='store_true', help='Train on already generated synthetic data')
parser.add_argument('--scenario6',action='store_true', help='Inference with LLM')
parser.add_argument('--generate_data',action='store_true')
parser.add_argument('--num_batches',type=int,default=5)
parser.add_argument('--data_dir',type=str,default='output_anli/')
parser.add_argument('--fb',type=str, default='None')
parser.add_argument('--save_dir',type=str,default='checkpoints/')
parser.add_argument('--loss',type=str,default='ce')
parser.add_argument('--alpha',type=float,default=1)
parser.add_argument('--beta',type=float,default=1)
parser.add_argument('--no_train',action='store_true')
parser.add_argument('--bot50',action='store_true')
parser.add_argument('--small_model',type=str,default='t5-base') #'google/t5-v1_1-base'

# Parse the argument
args = parser.parse_args()
print(args)

os.environ['TRANSFORMERS_CACHE'] ='~/.cache/huggingface/'
# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the paths to save the model
model_save_path = "checkpoints/"
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

if not os.path.exists(args.data_dir):
    os.mkdir(args.data_dir)

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)



with no_ssl_verification():
    # Initialize the T5 tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.small_model)
    model = T5ForConditionalGeneration.from_pretrained(args.small_model)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        # wrap the model with nn.DataParallel to train on multiple GPUs
        model = torch.nn.DataParallel(model)

max_length_input = 128
max_length_target=32
batch_size = args.bs
num_epochs = args.epochs

# Move the model to the appropriate device
model = model.to(device)

# Set the optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=args.lr)


if args.scenario0:
    
    files=glob.glob(args.data_dir+'*')
    print('####',len(files),'number of files exist in',args.data_dir,'####')
    batch_idx0=len(files)

    HF_TOKEN='token'
    model_llm = "meta-llama/Llama-2-7b-chat-hf"

    with no_ssl_verification():
        tokenizer_llm = AutoTokenizer.from_pretrained(model_llm,use_auth_token=HF_TOKEN)
        pipeline_llm = transformers.pipeline(
            "text-generation",
            model=model_llm,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=HF_TOKEN,
        )


    new_batch_fb=[]
    tot_batch=[]

    # test data:
    _,test_dataset=prepare_data(args.dataset,tokenizer=tokenizer,max_length_input=max_length_input,max_length_target=max_length_target)
    # Create data loaders for training and test sets
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    save_count=0
    for epoch in range(args.epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0
        num_samples = 0
        total_loss=0
        new_batch_fb_bad=[]
        new_batch_fb_good=[]
        new_batch=[]
        fb_good=[]
        
        for batch_idx in tqdm(range(args.num_batches)):
        
            # call chat-gpt to generate data
            new_batch=[]
            # all files that have been generated so far
            files=glob.glob(args.data_dir+'*')        
            if batch_idx%2==0: 
                fb_good=new_batch_fb_good
            else:
                fb_good=[]
            print('FB samples:\n',fb_good)
            
            if args.dataset=='anli':
                num_samples1=np.random.randint(2,4)
                num_samples3=np.random.randint(2,4)
                num_samples2=args.bs-num_samples1-num_samples3
                response1=llama2_anli(model=pipeline_llm,tokenizer=tokenizer_llm,num_samples=num_samples1,label=0,fb=fb_good)
                response2=llama2_anli(model=pipeline_llm,tokenizer=tokenizer_llm,num_samples=num_samples2,label=1,fb=fb_good)
                response3=llama2_anli(model=pipeline_llm,tokenizer=tokenizer_llm,num_samples=num_samples3,label=2,fb=fb_good)
                new_batch=batching_anli(response1,response2,response3,num_samples1,num_samples2,num_samples3)
       
            elif args.dataset=='qnli':
                new_batch=llama2_qnli(pipeline=pipeline_llm,tokenizer=tokenizer_llm,num_samples=args.bs,label=0,fb=fb_good)
            elif args.dataset=='rte':
                new_batch=llama2_rte(pipeline=pipeline_llm,tokenizer=tokenizer_llm,num_samples=args.bs,label=0,fb=fb_good)



                
            # save generated data
            with open(args.data_dir+str(batch_idx0+save_count)+'.json','w') as f:
                json.dump(new_batch,f)   
                save_count+=1                 

            # validation batch based feedback will be applid after the first epoch. First epoch is for warm-up.       
            if args.generate_data and args.fb=='val_ac':

                if batch_idx%25==0:

                    for file in tqdm(files):
                        with open(file,'r') as f:
                            batch_temp=json.load(f)
                        if args.dataset=='anli':
                            # Create the training dataset and data loader
                            train_dataset = ANLIDataset(tokenizer=tokenizer,dataset=batch_temp,max_length_input=max_length_input,max_length_target=max_length_target)
                        elif args.dataset=='qnli':
                            train_dataset = QNLIDataset(tokenizer=tokenizer,dataset=batch_temp,max_length_input=max_length_input,max_length_target=max_length_target)
                        elif args.dataset=='rte':
                            train_dataset = ANLIDataset(tokenizer=tokenizer,dataset=batch_temp,max_length_input=max_length_input,max_length_target=max_length_target,label_map=["entailment","not_entailment"])



                        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
                        

                        for batch in train_dataloader:
                            model.train()
                            inputs=batch['input_ids'].to(device)
                            targets = batch['label'].to(device)

                    
                            # Prepare attention mask
                            attention_mask = (inputs != tokenizer.pad_token_id).type(torch.float32)
                            decoder_attention_mask = (targets != tokenizer.pad_token_id).type(torch.float32)

                            optimizer.zero_grad()

                            #Forward pass
                            output=model(input_ids=inputs,attention_mask=attention_mask,labels=targets)
                            
            
                            loss = output.loss
                        
                            loss.backward()
                            optimizer.step()

                if args.dataset=='anli':
                    new_batch_val=[]
                    num_samples1=np.random.randint(2,4)
                    num_samples3=np.random.randint(2,4)
                    num_samples2=args.bs-num_samples1-num_samples3
                    response1=llama2_anli_val(num_samples=num_samples1,label=0,fb=new_batch)
                    response2=llama2_anli_val(num_samples=num_samples2,label=1,fb=new_batch)
                    response3=llama2_anli_val(num_samples=num_samples3,label=2,fb=new_batch)
                    new_batch_val=batching_anli(response1,response2,response3,num_samples1,num_samples2,num_samples3)
                elif args.dataset=='qnli':
                    new_batch_val=llama2_qnli_val(pipeline=pipeline_llm,tokenizer=tokenizer_llm,num_samples=args.bs,label=0,fb=fb_good)
                elif args.dataset=='rte':
                    new_batch_val=llama2_rte(pipeline=pipeline_llm,tokenizer=tokenizer_llm,num_samples=args.bs,label=0,fb=fb_good,val=True)


                # save generated data
                with open(args.data_dir+str(batch_idx0+save_count)+'.json','w') as f:
                    json.dump(new_batch_val,f)   
                    save_count+=1   

            
                # Create the validation dataset and data loader
                if args.dataset=='anli':
                    val_dataset = ANLIDataset(tokenizer=tokenizer,dataset=new_batch_val,max_length_input=max_length_input,max_length_target=max_length_target)
                elif args.dataset=='qnli':
                    val_dataset = QNLIDataset(tokenizer=tokenizer,dataset=new_batch_val,max_length_input=max_length_input,max_length_target=max_length_target)
                elif args.dataset in ['rte',]:
                    label_map=["entailment","not_entailment"]
                    val_dataset = ANLIDataset(tokenizer=tokenizer,dataset=new_batch_val,max_length_input=max_length_input,max_length_target=max_length_target,label_map=label_map)
                val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)

                with torch.no_grad():
                    for batch in val_dataloader:

                        model.eval()
                        inputs=batch['input_ids'].to(device)
                        target_text=batch['label_text']
                        target_ids=batch['label'].to(device)
                        attention_mask = (inputs != tokenizer.pad_token_id).type(torch.float32)
                        
                        # Generate output
                        outputs_ids = model.generate(inputs, max_length=max_length_target, num_beams=4, early_stopping=True)
                        outputs = model(input_ids=inputs,attention_mask=attention_mask,labels=target_ids)

                        logits = outputs.logits  # Get the logits for generated tokens


                        # Calculate negative log-likelihood loss for each sample
                        batch_size, seq_len, vocab_size = logits.shape
                        loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1), reduction='none')
                        energy = energy_score(logits.view(-1, vocab_size))
                        energy = energy.view(batch_size, seq_len)
                        energy = -energy.mean(dim=-1)
                        loss = loss.view(batch_size, seq_len)  # Reshape loss to match input shape
                        score = loss.mean(dim=-1)

            
                        idx  = energy.argsort()
                        print('Energy:',energy,'NLL:',score)

                        predicted_labels = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs_ids]
                        
                        
                        predicted_labels = [label.lower() for label in predicted_labels]
                        target_labels = [label.lower() for label in target_text]

                        total_correct = sum([1 for pred, target in zip(predicted_labels, target_labels) if pred == target])
                        total_samples = len(predicted_labels)

                    
                        accuracy = total_correct / total_samples
                        print('Accuracy:',accuracy)
                        new_batch_fb_good=[]
                        for ii in range(len(inputs)):
                            if ii in idx[args.bs*5//10:args.bs*8//10]:
                                dic={}
                                dic['sentence1']=batch['sentence1_text'][ii]
                                dic['sentence2']=batch['sentence2_text'][ii]
                                dic['label']=batch['label_text'][ii]
                                new_batch_fb_good.append(dic)


        print(f'Epoch {epoch}')

        model.eval()
    
        with torch.no_grad():
            total_loss = 0
            total_examples = 0
            total_correct=0
            total_samples=0
            for batch in tqdm(test_dataloader):
                input_ids = batch['input_ids'].to(device)
                target_text = batch['label_text']
            
                # Generate output
                output_ids = model.generate(input_ids, max_length=max_length_target, num_beams=4, early_stopping=True)
                predicted_labels = [tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]
                                                
              
                predicted_labels = [label.lower() for label in predicted_labels]
                target_labels = [label.lower() for label in target_text]
                total_correct += sum([1 for pred, target in zip(predicted_labels, target_labels) if pred == target])
                total_samples += len(predicted_labels)

            
            accuracy = total_correct / total_samples
            print('Accuracy:',accuracy)



elif args.scenario4:
    
    files=glob.glob(args.data_dir+'*')

    new_batch=[]
    for idx in range(args.num_batches):
        file=args.data_dir+str(idx)+'.json'
        with open(file,'r') as f:
            new_file=json.load(f)
            for ll in range(len(new_file)):
                new_batch.append(new_file[ll])
  
    print('####',len(files),'number of files exist in',args.data_dir,'####')
    files_train=files[:args.num_batches]
    files_test=files[args.num_batches*9//10:args.num_batches]
    batch_idx0=len(files)
    batch_count=0
    new_batch_fb=[]
    tot_batch=[]

    # test data:
    _,test_dataset=prepare_data(args.dataset,tokenizer=tokenizer,max_length_input=max_length_input,max_length_target=max_length_target)
    # Create data loaders for training and test sets
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
    # Create the training dataset and data loader
    if args.dataset in ['anli','mnli','rte','wnli','mrpc']:
        train_dataset = ANLIDataset(tokenizer=tokenizer,dataset=new_batch,max_length_input=max_length_input,max_length_target=max_length_target,task=args.dataset)
    elif args.dataset=='qnli':
        train_dataset = QNLIDataset(tokenizer=tokenizer,dataset=new_batch,max_length_input=max_length_input,max_length_target=max_length_target,task=args.dataset)
    elif args.dataset=='cola':
        train_dataset = COLADataset(tokenizer=tokenizer,dataset=new_batch,max_length_input=max_length_input,max_length_target=max_length_target)
    elif args.dataset=='mtc2':
        train_dataset = mtc_dataset(tokenizer=tokenizer,dataset=new_batch,max_length_input=max_length_input,max_length_target=max_length_target)
 

    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False)    

    
    # ## train on OOD generated data:
    for epoch in range(args.epochs):
        model.train()
        

        for batch in tqdm(train_dataloader):

        
            inputs=batch['input_ids'].to(device)
            targets = batch['label'].to(device)
    
            # Prepare attention mask
            attention_mask = (inputs != tokenizer.pad_token_id).type(torch.float32)
            decoder_attention_mask = (targets != tokenizer.pad_token_id).type(torch.float32)
          
            optimizer.zero_grad()

            #Forward pass
            output=model(input_ids=inputs,attention_mask=attention_mask,labels=targets)

            # Calculate symmetric cross entropy loss for each sample
            logits = output.logits 
            batch_size, seq_len, vocab_size = logits.shape

            energy = energy_score(logits.view(-1, vocab_size))
            energy = energy.view(batch_size, seq_len)
            energy = -energy.mean(dim=-1)
            idx  = energy.argsort()

            logits_filtered=logits[idx[4:]]
            targets_filtered=targets[idx[4:]]

            if args.bot50:
                logits1=logits_filtered
                targets1=targets_filtered
            else:
                logits1=logits
                targets1=targets                


            if args.loss=='sce':
                criterion=SCELoss(alpha=args.alpha,beta=args.beta,num_classes=vocab_size)
                loss1=criterion.forward(logits1.view(-1, vocab_size), targets1.view(-1))
            else:
                loss1=output.loss


            
            loss = loss1.mean()
        
            loss.backward()
            optimizer.step()     
    
            
        # Validate the model
        model.eval()
    
        with torch.no_grad():
            total_loss = 0
            total_examples = 0
            total_correct=0
            total_samples=0
            for batch in tqdm(test_dataloader):
                input_ids = batch['input_ids'].to(device)
                target_text = batch['label_text']
 
                # Generate output
                if torch.cuda.device_count() > 1:
                    output_ids = model.module.generate(input_ids, max_length=max_length_input, num_beams=4, early_stopping=True)
                else:
                    output_ids = model.generate(input_ids, max_length=max_length_input, num_beams=4, early_stopping=True)
                predicted_labels = [tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]
                
                
                predicted_labels = [label.lower() for label in predicted_labels]
                target_labels = [label.lower() for label in target_text]



                total_correct += sum([1 for pred, target in zip(predicted_labels, target_labels) if pred == target])
                total_samples += len(predicted_labels)

           
            accuracy = total_correct / total_samples
            print('Accuracy:',accuracy)

elif args.scenario6:
    from transformers import logging

    # Set the verbosity to warning
    logging.set_verbosity_warning()
    # Set the verbosity to error
    logging.set_verbosity_error()

    HF_TOKEN='token'
    model_llm = "meta-llama/Llama-2-7b-chat-hf"

    with no_ssl_verification():
        tokenizer_llm = AutoTokenizer.from_pretrained(model_llm,use_auth_token=HF_TOKEN)
        pipeline_llm = transformers.pipeline(
            "text-generation",
            model=model_llm,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=HF_TOKEN,
        )


    # test data:
    _,test_dataset=prepare_data(args.dataset,tokenizer=tokenizer,max_length_input=max_length_input,max_length_target=max_length_target)
    # Create data loaders for training and test sets
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    

    total_loss = 0
    total_examples = 0
    total_correct=0
    total_samples=0
    for batch in tqdm(test_dataloader):
        target_text = batch['label_text'][0]

        
        if args.dataset in ['anli','mnli']:
            input_text= "\nPremise: "+ batch['sentence1_text'][0]+" \nHypothesis: "+batch['sentence2_text'][0]
            prompt= f"You are a helpful assistant. You only reply once as 'Assistant' do not pretend to be a 'User'. Take a deep breath and think step-by-step. Below are 3 examples of an natural language inference dataset. Samples include a 'hypothesis' and a 'premise'. The label of sample is: 1. 'entailment' if premise entail hypothesis, 2. 'neutral' if premise neither entail nor contradict hypothesis. 3.'contradiction' if premise contradict hypothesis \n\n 1. \n\n Premise: The Parma trolleybus system (Italian: 'Rete filoviaria di Parma' ) forms part of the public transport network of the city and 'comune' of Parma, in the region of Emilia-Romagna, northern Italy. In operation since 1953, the system presently comprises four urban routes. \n\n Hypothesis: The trolleybus system has over 2 urban routes \n\n Label: entailment \n\n 2. \n\n Premise: The Centralia Massacre was an incident during the American Civil War in which twenty-four unarmed Union soldiers were captured and executed at Centralia, Missouri on September 27, 1864 by the pro-Confederate guerrilla leader William T. Anderson. Future outlaw Jesse James was among the guerrillas. \n\n Hypothesis: Jesse James was a guerrilla in the Union army during the American Civil War. \n\n Label: contradiction \n\n 3. \n\n Premise: Alexandra Lendon Bastedo (9 March 1946 – 12 January 2014) was a British actress, best known for her role as secret agent Sharron Macready in the 1968 British espionage/science fiction adventure series 'The Champions'. She has been cited as a sex symbol of the 1960s and 1970s. Bastedo was a vegetarian and animal welfare advocate.\n\n Hypothesis:Sharron Macready was a popular character through the 1980 s. \n\n Label: neutral. \n\n The above are samples of natural language inference data. What is the label of the following example: {input_text} . \n Reply very short. \nAssistant: The label of the above example is"

        elif args.dataset in ['rte','wnli']:
            input_text= "\nPremise: "+ batch['sentence1_text'][0]+" \nHypothesis: "+batch['sentence2_text'][0]
            prompt= f"You are a helpful 'Assistant'. You only reply once as 'Assistant'. Do not pretend to be a 'User'. Below are three samples of RTE dataset. Samples include a 'Sentence1' and a 'Sentence2'. The label of sample is 'entailment' if the answer of the 'Sentence1' entails 'Sentence2' and 'not_entailment' if 'Sentence1' does not entail 'Sentence2'. \n\n 1. \nSentence1: No Weapons of Mass Destruction Found in Iraq Yet. \nSentence2: Weapons of Mass Destruction Found in Iraq. \nLabel: not_entailment \n\n 2. \nSentence1: A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI. \nSentence2: Pope Benedict XVI is the new leader of the Roman Catholic Church.  \nLabel: entailment \n\n 3. \nSentence1:Herceptin was already approved to treat the sickest breast cancer patients, and the company said, Monday, it will discuss with federal regulators the possibility of prescribing the drug for more breast cancer patients. \nSentence2:Herceptin can be used to treat breast cancer. \nLabel: entailment \n\n The above are three samples of RTE data. \n What is the label of the following example: {input_text} \n Reply very short. \nAssistant: The label of the above example is"

        elif args.dataset=='mrpc':
            input_text= "\nSentence 1: "+ batch['sentence1_text'][0]+" \nSentence 2: "+batch['sentence2_text'][0]
            prompt= f"You are a helpful 'Assistant'. You only reply once as 'Assistant'. Do not pretend to be a 'User'. Below are three samples of MRPC dataset. Samples include a 'Sentence 1' and a 'Sentence 2'. The label of sample is 'equivalent' if the 'Sentence 1' and the 'Sentence 2' are paraphrases of each other. The label is 'not_equivalent' if 'Sentence 1' and 'Sentence 2' are not semantically equivalent. \n\n 1. \nSentence 1: Amrozi accused his brother , whom he called  the witness , of deliberately distorting his evidence . \nSentence 2: Referring to him as only the witness , Amrozi accused his brother of deliberately distorting his evidence . \nLabel: equivalent \n\n 2. \nSentence 1: Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion . \nSentence 2: Yucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 . \nLabel: not_equivalent \n\n 3. \nSentence 1: Around 0335 GMT , Tab shares were up 19 cents , or 4.4 % , at A $ 4.56 , having earlier set a record high of A $ 4.57 . \nSentence 2: Tab shares jumped 20 cents , or 4.6 % , to set a record closing high at A $ 4.57 . \nLabel: not_equivalent \n\n The above are three samples of MRPC data. \n What is the label of the following example: {input_text} \n Reply very short. \nAssistant: The label of the above example is"

        elif args.dataset=='qnli':
            input_text= "\nSentence: "+ batch['sentence_text'][0]+" \nQuestion: "+batch['question_text'][0]
            prompt= f"You are a helpful 'Assistant'. You only reply once as 'Assistant'. Do not pretend to be a 'User'. Below are three samples of QNLI dataset. Samples include a 'Question' and a 'Sentence'. The label of sample is 1.'entailment' if the answer of the 'Question' is in the 'Sentence' and 2.'not_entailment' if the answer of the 'Question' is not in the 'Sentence'. \n\n 1. \nSentence: He must do this by collecting the multiple Tears of Light; once all the Tears of Light are collected for one area, he restores that area's Light Spirit. \nQuestion: What does Link have to gather in order to complete each area? \nLabel: entailment \n\n 2. \nSentence:Prior to this time congressional parties were often relatively disorganized, so it was not always evident who functioned as the opposition floor leader. \nQuestion: Why was minority leader position created?. \nLabel: entailment \n\n 3. \nSentence:This view is shared by other researchers who argue that the ancestors of the American Indians were the first to separate from the great Asian population in the Middle Paleolithic. \nQuestion:Who have studies of the mtDNA of Turkic-speaking peoples shown they're closest to genetically? \nLabel: not_entailment \n\n The above are three samples of QNLI data. \n What is the label of the following example: {input_text} \n Reply very short. \nAssistant: The label of the above example is"
       
            


        # labels = batch['label'].to(device)
        with no_ssl_verification():
            predicted_label= pipeline_llm(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer_llm.eos_token_id,
            max_length=2000,)
            response=predicted_label[0]['generated_text']
            idx0=response.find('Assistant:')
            response=response[idx0:].replace('\n','').strip()

            if args.dataset=='anli':
                if 'entailment' in response:
                    predicted_label='entailment'
                elif 'neutral' in response:
                    predicted_label='neutral'
                else:
                    predicted_label='contradiction'
            elif args.dataset in ['rte','qnli']:
                if 'not_entailment' in response:
                    predicted_label='not_entailment'
                else:
                    predicted_label='entailment'

            elif args.dataset=='mrpc':
                if 'not_equivalent' in response:
                    predicted_label='not_equivalent'
                else:
                    predicted_label='equivalent'

            if predicted_label == target_text:
                total_correct += 1
            total_samples += 1
            accuracy = total_correct / total_samples
            print('Accuracy:',accuracy)

    
    accuracy = total_correct / total_samples
    print('Accuracy:',accuracy)


