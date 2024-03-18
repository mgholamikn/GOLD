# # scenario 1:
# python T5_squad.py --scenario1 --data_dir generated_data/svamp/ --num_batches 375 --epochs 1 --fb val_ac --dataset svamp   

# # scenario 0:
# python T5_squad.py --scenario0 --epochs 20 --dataset svamp 

# # scenario 4: 
# python T5_squad.py --scenario4 --data_dir generated_data/svamp/ --num_batches 375 --epochs 50 --dataset svamp  --loss sce --alpha 0.1  

# # scenario 6: 
# python T5_squad.py --scenario6 --dataset svamp 

from transformers import T5Tokenizer, T5ForConditionalGeneration,  AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from utils.ssl import no_ssl_verification
from tqdm import tqdm
from torch.nn import DataParallel
import glob
import json
from utils.datasets import *
import argparse
from utils.chatgpt import *
from utils.batching import batching_squad
from utils.common import energy_score, eval_equation, SCELoss
import os
import time
import transformers
import evaluate

# import os
os.environ['CURL_CA_BUNDLE'] = ''

parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--dataset',type=str,default='squad')
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--lr',type=float,default=5e-5)
parser.add_argument('--bs',type=int,default=8)
parser.add_argument('--llm',type=str,default='llama2')
parser.add_argument('--pd',type=float,default=100,help='percent of real data to be used when in the case of train_real or scenario 2')
parser.add_argument('--scenario0',action='store_true',help='Fine-tune on real data')
parser.add_argument('--scenario1',action='store_true', help='Generate Data without FB')
parser.add_argument('--scenario4',action='store_true', help='Train on already Generated Data')
parser.add_argument('--scenario6',action='store_true', help='Inference with LLM')
parser.add_argument('--num_batches',type=int,default=5)
parser.add_argument('--data_dir',type=str,default='output_anli/')
parser.add_argument('--small_model',type=str,default='t5-base')
parser.add_argument('--alpha',type=float,default=1)
parser.add_argument('--beta',type=float,default=1)
parser.add_argument('--fb_rate',type=float,default=0.5)
parser.add_argument('--fb',type=str, default='None')
parser.add_argument('--save_dir',type=str,default='checkpoints/T5/')
parser.add_argument('--loss',type=str,default='ce')
parser.add_argument('--seed',type=int,default=44)
# Parse the argument
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
if not os.path.exists(args.data_dir):
    os.mkdir(args.data_dir)

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

if args.dataset in ['temporal','mtc']:
    chrf = evaluate.load("chrf")
if args.dataset in ['mtc','nl4opt']:
    from datasets import load_metric
    # Load the ROUGE metric
    rouge = load_metric('rouge')


with no_ssl_verification():
    # Instantiate a T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(args.small_model)
    tokenizer = T5Tokenizer.from_pretrained(args.small_model)


# Set the device to run on: GPU if available, else CPU
# If you want to use specific GPUs (e.g., using 2 out of 4 GPUs):
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # specify the GPU 

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    # wrap the model with nn.DataParallel to train on multiple GPUs
    model = torch.nn.DataParallel(model)
model = model.to(device)
# Set the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

# Set the batch size and number of epochs
batch_size = args.bs
epochs = args.epochs

if args.dataset=='nl4opt':
    max_length_input=512
    max_length_target=256
elif args.dataset=='mtc':
    max_length_input=256
    max_length_target=128   
elif args.dataset=='mtc2':
    max_length_input=256
    max_length_target=32
elif args.dataset in ['squad','adversarialQA']:
    max_length_input=256
    max_length_target=32
else:
    max_length_input=128
    max_length_target=32




if args.scenario0:
    train_dataset,test_dataset=prepare_data(args.dataset,tokenizer=tokenizer,max_length_input=max_length_input,max_length_target=max_length_target,percent_data=args.pd)

    # Create data loaders for training and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        
        # Keep track of the training loss
        train_loss = 0
        
        # Iterate over the training data
        for batch in tqdm(train_dataloader):
            # Move the data to the device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
            
            # Compute the loss
            loss = outputs.loss
            loss=loss.mean() 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update the training loss
            train_loss += loss.item()
      
        
        # Compute the average training loss for this epoch
        train_loss /= len(train_dataloader)

            
        # Set the model to evaluation mode
        model.eval()
        
        # Keep track of the validation loss and accuracy
        valid_loss = 0
        exact_match_count=0
        total_examples=0

        pred_all=[]
        ref_all=[]        
        # Iterate over the validation data
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                # Move the data to the device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
                
                # Compute the loss and accuracy
                loss = outputs[0].mean()

                
                # Move the generated output to CPU and decode it using the tokenizer
                generated_ids = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                predicted_labels = [tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]

                # Convert target IDs to text
                target_text = tokenizer.batch_decode(batch['label'], skip_special_tokens=True)
                
                if args.dataset in ['squad']:
                    # Calculate exact match for each example in the batch
                    for pred, target in zip(predicted_labels, target_text):
                        if pred == target:
                            exact_match_count += 1
                        total_examples += 1
                elif args.dataset in ['nl4opt']:
                    pred_all+=predicted_labels
                    ref_all+=target_text                    
                # Update the validation loss
                valid_loss += loss.item()
            

        if args.dataset in ['mtc','nl4opt']:
            results=rouge.compute(predictions=pred_all, references=ref_all)
            score=results["rougeL"].mid.fmeasure
            print(f'Epoch: {epoch+1}/{epochs}... ROUGE-L: {score:.3f}')
        
        elif args.dataset in ['squad']:
            # Compute the average validation loss and exact match rate for this epoch
            exact_match_rate = exact_match_count / total_examples
            # Print epoch results
            print(f'Epoch: {epoch+1}/{epochs}... Exact Match: {exact_match_rate:.3f}')

elif args.scenario1:

    HF_TOKEN='token'
    model_llm = "meta-llama/Llama-2-7b-chat-hf"

    with no_ssl_verification():
        tokenizer_llm = AutoTokenizer.from_pretrained(model_llm,token=HF_TOKEN)
        pipeline_llm = transformers.pipeline(
            "text-generation",
            model=model_llm,
            torch_dtype=torch.float16,
            device_map="balanced_low_0",
            token=HF_TOKEN,
        )

    files=glob.glob(args.data_dir+'*')
    print('####',len(files),'number of files exist in',args.data_dir,'####')
    batch_idx0=len(files)

    new_batch_fb=[]
    tot_batch=[]


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
        exact_match_count=0
        
        for batch_idx in tqdm(range(args.num_batches)):
            # all files that have been generated so far
            files=glob.glob(args.data_dir+'*')
            
            # 
            new_batch=[]
            
            if batch_idx%np.round(1/args.fb_rate)==0: 
                fb_good=new_batch_fb_good
            else:
                fb_good=[]
            print('FB samples:\n',fb_good)
            
            new_batch=[]
            
            if args.dataset in ['squad',]:
                new_batch=llama2_squad(pipeline=pipeline_llm,tokenizer=tokenizer_llm,num_samples=args.bs,label=0,fb=fb_good)
            elif args.dataset=='svamp':
                new_batch=llama2_svamp(pipeline=pipeline_llm,tokenizer=tokenizer_llm,num_samples=args.bs,label=0,fb=fb_good)
            elif args.dataset=='nl4opt':
                new_batch=llama2_nl4opt(pipeline=pipeline_llm,tokenizer=tokenizer_llm,num_samples=args.bs,label=0,fb=fb_good)
            elif args.dataset=='mtc':
                new_batch=llama2_mtc(pipeline=pipeline_llm,tokenizer=tokenizer_llm,num_samples=args.bs,label=0,fb=fb_good,temp_setting='scheduler')
                
            # save generated data
            files=glob.glob(args.data_dir+'*')
            with open(args.data_dir+str(len(files))+'.json','w') as f:
                json.dump(new_batch,f)   
                save_count+=1                   


            # validation batch based feedback will be applid after the first epoch. First epoch is for warm-up.
                  
            if args.fb=='val_ac' and batch_idx%np.round(1/args.fb_rate)==0:
                new_batch_val=[]
                # # print('train batch:\n',new_batch)
                
                if batch_idx%25==0 and batch_idx>0:
                    for file in tqdm(files):
                        with open(file,'r') as f:
                            batch_temp=json.load(f)
                        # Create the training dataset and data loader
                        if args.dataset in ['squad','adversarialQA']:
                            train_dataset = squad_dataset(tokenizer,batch_temp,max_length_input=max_length_input,max_length_target=max_length_target)
                        elif args.dataset=='svamp':
                            train_dataset = svamp_dataset(tokenizer,batch_temp,max_length_input=max_length_input,max_length_target=max_length_target)
                        elif args.dataset=='nl4opt':
                            train_dataset = nl4opt_dataset(tokenizer,batch_temp,max_length_input=max_length_input,max_length_target=max_length_target)
                        elif args.dataset=='mtc':
                            train_dataset = mtc_dataset(tokenizer,batch_temp,max_length_input=max_length_input,max_length_target=max_length_target)

                        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
                        
            
                        for batch in train_dataloader:
                            model.train()

                            inputs=batch['input_ids'].to(device)
                            targets = batch['label'].to(device)
                            attention_mask = (inputs != tokenizer.pad_token_id).type(torch.float32)

                            
                            # Forward pass
                            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=targets)
                            
                            optimizer.zero_grad()
                            # Compute the loss
                            loss = outputs.loss
                            loss=loss.mean() 
                            loss.backward()
                            optimizer.step()
                            
                            
                            # Update the training loss
                            train_loss += loss.item()


                if args.dataset in ['squad']:
                    new_batch_val=llama2_squad(pipeline=pipeline_llm,tokenizer=tokenizer_llm,num_samples=args.bs,label=0,fb=new_batch[:2],val=True)
                elif args.dataset in ['svamp']:
                    new_batch_val=llama2_svamp(pipeline=pipeline_llm,tokenizer=tokenizer_llm,num_samples=args.bs,label=0,fb=new_batch[:2],val=True)
                elif args.dataset in ['nl4opt']:
                    new_batch_val=llama2_nl4opt(pipeline=pipeline_llm,tokenizer=tokenizer_llm,num_samples=args.bs,label=0,fb=new_batch[:2],val=True)
                elif args.dataset in ['mtc']:
                    new_batch_val=llama2_mtc(pipeline=pipeline_llm,tokenizer=tokenizer_llm,num_samples=args.bs,label=0,fb=new_batch[:2],val=True,temp_setting='scheduler')

                # save generated data
                files=glob.glob(args.data_dir+'*')
                with open(args.data_dir+str(len(files))+'.json','w') as f:
                    json.dump(new_batch_val,f)   
                    save_count+=1  
            
                # Create the validation dataset and data loader
                if args.dataset in ['squad']:
                    val_dataset = squad_dataset(tokenizer,new_batch_val,max_length_input=max_length_input,max_length_target=max_length_target)
                elif args.dataset=='svamp':
                    val_dataset = svamp_dataset(tokenizer,new_batch_val,max_length_input=max_length_input,max_length_target=max_length_target)
                elif args.dataset=='nl4opt':
                    val_dataset = nl4opt_dataset(tokenizer,new_batch_val,max_length_input=max_length_input,max_length_target=max_length_target)
                elif args.dataset=='mtc':
                    val_dataset = mtc_dataset(tokenizer,new_batch_val,max_length_input=max_length_input,max_length_target=max_length_target)

                val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)

                with torch.no_grad():
                    for batch in val_dataloader:

                        model.eval()
                        inputs=batch['input_ids'].to(device)
                        target_text=batch['label_text']
                        target_ids=batch['label'].to(device)
                        attention_mask = (inputs != tokenizer.pad_token_id).type(torch.float32)
                        
                        # Forward pass
                        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=target_ids)

                        
                        # Move the generated output to CPU and decode it using the tokenizer
                        generated_ids = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                        logits = outputs.logits  # Get the logits for generated tokens


                        batch_size, seq_len, vocab_size = logits.shape
                        loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1), reduction='none')
                        energy = energy_score(logits.view(-1, vocab_size))
                        energy = energy.view(batch_size, seq_len)
                        energy = -energy.mean(dim=-1)
                        loss = loss.view(batch_size, seq_len)  # Reshape loss to match input shape
                        score = loss.mean(dim=-1)
                        print(score,energy)
                        idx  = energy.argsort()

                        # Move the generated output to CPU and decode it using the tokenizer
                        generated_ids = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                        # Convert target IDs to text
                        target_text = tokenizer.batch_decode(batch['label'], skip_special_tokens=True)
                        
                        if args.dataset in ['svamp','squad']:
                            # Calculate exact match for each example in the batch
                            exact_match_count=0
                            for pred, target in zip(generated_text, target_text):
                                if pred == target:
                                    exact_match_count += 1

                            print('Accuracy:',exact_match_count/len(generated_text))
                        new_batch_fb_good=[]
                        for ii in range(len(batch['input_ids'])):
                            if ii in idx[args.bs*5//10:args.bs*8//10]:
                                dic={}
                                if args.dataset=='squad':
                                    dic['context']=batch['context'][ii]
                                    dic['question']=batch['question'][ii]
                                    dic['answer']=batch['label_text'][ii]
                                    new_batch_fb_good.append(dic)   
                                elif args.dataset=='svamp':
                                    dic['Body']=batch['body'][ii]
                                    dic['Question']=batch['question'][ii]
                                    dic['Equation']=batch['label_text'][ii]
                                    dic['Answer']=batch['answer'][ii]
                                elif args.dataset=='nl4opt':
                                    dic['problem']=batch['problem'][ii]
                                    dic['formulation']=batch['formulation'][ii]
                                    new_batch_fb_good.append(dic) 
                                elif args.dataset=='mtc':
                                    dic['Dialogue']=batch['Dialogue'][ii]
                                    dic['Note']=batch['label_text'][ii]
                                    new_batch_fb_good.append(dic) 


        avg_loss = total_loss / (len(train_dataloader)*args.num_batches)
        print(f'Epoch {epoch} - Avg. Loss: {avg_loss}')
 


elif args.scenario4:

    new_batch=[]

    files=glob.glob(args.data_dir+'*')
    print('####',len(files),'number of files exist in',args.data_dir,'####')

    files=glob.glob(args.data_dir+'*')

    new_batch=[]
    # for file in files[:args.num_batches]:
    for idx in range(args.num_batches):
        file=args.data_dir+str(idx)+'.json'
        with open(file,'r') as f:
            new_file=json.load(f)
            for ll in range(len(new_file)):
                new_batch.append(new_file[ll])

    batch_idx0=len(files)
    batch_count=0
    train_loss=0
    new_batch_fb=[]
    tot_batch=[]

    # test data:
    _,test_dataset=prepare_data(args.dataset,tokenizer=tokenizer,max_length_input=max_length_input,max_length_target=max_length_target)
    # Create data loaders for training and test sets
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
    
    
    # model.train()
    ## train on OOD generated data:
    for epoch in range(args.epochs):
        model.train()


        # Create the training dataset and data loader
        if args.dataset in ['squad','adversarialQA']:
            train_dataset = squad_dataset(tokenizer,new_batch,max_length_input=max_length_input,max_length_target=max_length_target)
        elif args.dataset=='svamp':
            train_dataset = svamp_dataset(tokenizer,new_batch,max_length_input=max_length_input,max_length_target=max_length_target)
        elif args.dataset=='temporal':
            train_dataset = temporal_dataset(tokenizer,new_batch,max_length_input=max_length_input,max_length_target=max_length_target)
        elif args.dataset=='mtc':
            train_dataset = mtc_dataset(tokenizer,new_batch,max_length_input=max_length_input,max_length_target=max_length_target)
        elif args.dataset=='nl4opt':
            train_dataset = nl4opt_dataset(tokenizer,new_batch,max_length_input=max_length_input,max_length_target=max_length_target)

        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        

        for batch in tqdm(train_dataloader):
            

            input_ids=batch['input_ids'].to(device)
            attention_mask=batch['attention_mask'].to(device)
            labels=batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            optimizer.zero_grad()

            # Calculate symmetric cross entropy loss for each sample
            logits = outputs.logits 
            batch_size, seq_len, vocab_size = logits.shape
            if args.loss=='sce':
                criterion=SCELoss(alpha=args.alpha,beta=args.beta,num_classes=vocab_size)
                loss1=criterion.forward(logits.view(-1, vocab_size), labels.view(-1))
            else:
                loss1 = outputs.loss
            loss = loss1.mean()

 
            loss.backward()
            optimizer.step()
            
            
            # Update the training loss
            train_loss += loss.item()  
        
            
        # Set the model to evaluation mode
        model.eval()
        
        # Keep track of the validation loss and accuracy
        valid_loss = 0
        exact_match_count=0
        total_examples=0

        pred_all=[]
        ref_all=[]

        # Iterate over the validation data
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                # Move the data to the device
                input_ids=batch['input_ids'].to(device)
                attention_mask=batch['attention_mask'].to(device)
                labels=batch['label'].to(device)
                target_text=batch['label_text']
                
                if torch.cuda.device_count() > 1:
                    output_ids = model.module.generate(input_ids, max_length=max_length_input, num_beams=4, early_stopping=True)
                else:
                    output_ids = model.generate(input_ids, max_length=max_length_input, num_beams=4, early_stopping=True)
                predicted_labels = [tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]
                

                
                if args.dataset in ['squad','adversarialQA']:
                    # Calculate exact match for each example in the batch
                    for ii in range(len(predicted_labels)):
         
                        if predicted_labels[ii] in target_text[ii].split('\n'):
                            exact_match_count += 1 
                    
                elif args.dataset in ['svamp']:
                    predicted_labels = [eval_equation(pred) for pred in predicted_labels]
                    target_text = [eval_equation(gt) for gt in target_text]
                    # print(predicted_labels,target_text)
                    exact_match_count += sum([1 for pred, target in zip(predicted_labels, target_text) if pred==target])
                elif args.dataset in ["temporal","mtc","nl4opt"]:
                    print('pred:\n',predicted_labels)
                    print('labels:\n',target_text)
                    pred_all+=predicted_labels
                    ref_all+=target_text
                
                total_examples += len(predicted_labels)
                


        if args.dataset in ['temporal']:
            result=chrf.compute(predictions=pred_all, references=ref_all)
            score=result["score"]
        elif args.dataset in ['mtc','nl4opt']:
            results=rouge.compute(predictions=pred_all, references=ref_all)
            score=results["rougeL"].mid.fmeasure
        else:
            # Compute the average validation loss and exact match rate for this epoch
            valid_loss /= len(test_dataloader)
            exact_match_rate = exact_match_count / total_examples
        
        if args.dataset in ['temporal',]:
            print(f'Epoch: {epoch+1}/{epochs}... ChrF++: {score:.3f}')
        elif args.dataset in ['mtc','nl4opt']:
            print(f'Epoch: {epoch+1}/{epochs}... ROUGE-L: {score:.3f}')
        # Print epoch results
        elif args.dataset in ['squad','adversarialQA']:
            print(f'Epoch: {epoch+1}/{epochs}... Exact Match: {exact_match_rate:.3f}')
        else:
            print(f'Epoch: {epoch+1}/{epochs}... Accuracy: {exact_match_rate:.3f}')


elif args.scenario6:

    def remove_letters(input_string):
        return ''.join(char for char in input_string if char.isdigit() or char in ('(', ')','-','+','/','.','*'))

    from transformers import logging

    # Set the verbosity to warning
    logging.set_verbosity_warning()
    # Set the verbosity to error
    logging.set_verbosity_error()

    HF_TOKEN='token'
    model_llm = "meta-llama/Llama-2-7b-chat-hf"

    with no_ssl_verification():
        tokenizer_llm = AutoTokenizer.from_pretrained(model_llm,token=HF_TOKEN)
        pipeline_llm = transformers.pipeline(
            "text-generation",
            model=model_llm,
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN,
            temperature=0.1,
        )


    # test data:
    _,test_dataset=prepare_data(args.dataset,tokenizer=tokenizer,max_length_input=max_length_input,max_length_target=max_length_target)
    # Create data loaders for training and test sets
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    

    total_loss = 0
    total_examples = 0
    total_correct=0
    total_samples=0
    pred_all=[]
    ref_all=[]
    for batch in tqdm(test_dataloader):
        target_text = batch['label_text'][0]

        
        if args.dataset in ['squad','adversarialQA']:
            input_text= "\ncontext: "+ batch['context'][0]+" \nquestion: "+batch['question'][0]
            prompt= f"\n\n 1. context: Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend 'Venite Ad Me Omnes'. Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary. question:To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France? answer:Saint Bernadette Soubirous \n\n 2. context: As at most other universities, Notre Dame's students run a number of news media outlets. The nine student-run outlets include three newspapers, both a radio and television station, and several magazines and journals. Begun as a one-page journal in September 1876, the Scholastic magazine is issued twice monthly and claims to be the oldest continuous collegiate publication in the United States. The other magazine, The Juggler, is released twice a year and focuses on student literature and artwork. The Dome yearbook is published annually. The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news, and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent publication and does not have a faculty advisor or any editorial oversight from the University. In 1987, when some students believed that The Observer began to show a conservative bias, a liberal newspaper, Common Sense was published. Likewise, in 2003, when other students believed that the paper showed a liberal bias, the conservative paper Irish Rover went into production. Neither paper is published as often as The Observer; however, all three are distributed to all students. Finally, in Spring 2008 an undergraduate journal for political science research, Beyond Politics, made its debut. question: In what year did the student paper Common Sense begin publication at Notre Dame? answer: 1987 \n\n 3. context: The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study – aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering – with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively. question: How many departments are within the Stinson-Remick Hall of Engineering? answer: fiven \n\n The above are samples of SQUAD dataset. What is the label of the following example: {input_text} . \nAssistant: The label of the above example is:"

        elif args.dataset in ['svamp']:
            input_text= "\Body: "+ batch['body'][0]+" \nQuestion: "+batch['question'][0]
            prompt= f"You are a helpful assistant. You only reply once as 'Assistant' do not pretend to be a 'User'. Take a deep breath and think step-by-step. Below are 3 examples of SVAMP dataset. Samples include a 'Body' which is explains a simple math problem and a 'Quesiton' which ask a question from the 'Body'. \n\n 1. \nBody:There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups. \nQuestion: How big is each group of bananas? \nEquation:( 290.0 / 2.0 ) \n\n 2. \n Body: Marco and his dad went strawberry picking. Marco's dad's strawberries weighed 11 pounds. If together their strawberries weighed 30 pounds. \nQuestion: How much did Marco's strawberries weigh? \nEquation: ( 30.0 - 11.0 ) \n\n 3. \nBody: Edward spent $ 6 to buy 2 books each book costing him the same amount of money. Now he has $ 12. \nQuestion: How much did each book cost? \nEquation: ( 6.0 / 2.0 ) \n\n The above are samples of  SVAMP data. {input_text} \n What is the Equation?. Assistant: Sure! here is the Equation. I do not add = in the equation. I have made my answer short and removed any text. Equation:"

        elif args.dataset in ['nl4opt']:
            print('###',batch['problem'])
            input_text="Problem: "+ batch['problem'][0]
            prompt=f"You are a helpful assistant. You only reply once as 'Assistant' do not pretend to be a 'User'. The above are examples of NL4OPT dataseta and below are three samples of 'NL4OPT' dataset. The samples have a 'problem' which is a linear optimization problem and a 'formulation' which is the formulation of the optimization problem. \n\n1. \nproblem: A hotel employs cleaners and receptionists. Cleaners earn $500 per week and receptionists earn $350 per week. The hotel requires a minimum of 100 workers of whom at least 20 must be receptionists. To keep the hotel clean and running smoothly, the number of receptionists should be at least a third of the number of cleaners. The hotel wants to keep the weekly wage bill below $30000. Formulate a LP to minimize the wage bill. \nformulation: \nVariables:\n x: sled dogs, y: trucks\nObjective Function:\nmaximize\n100.00 * x + 300.00 * y\nConstraints:\n50.00 * x + 100.00 * y \u2264 1000.00\n1.00 * x  \u2264 1.00 * y \n\n2. \nproblem: An office supply company makes two types of printers: color printers and black and white printers. Different sections of the factory with different teams produce each printer. The color printer team can produce at most 20 color printers per day while the black and white printer team can produce at most 30 black and white printers per day. Both teams require use of the same paper tray installing machine and this machine can make at most 35 printers of either type each day. Color printers generate a profit of $200 per printer while black and white printers generate a profit of $70 per printer. How many of each printer should be made to maximize the company's profit? \nformulation: \nVariables:\n x: color printers, y: black and white printers\nObjective Function:\nmaximize\n200.00 * x + 70.00 * y\nConstraints:\n1.00 * x \u2264 20.00\n1.00 * y \u2264 30.00\n1.00 * x + 1.00 * y \u2264 35.00 \n\n3. \nproblem: An accounting firm has senior accountants earning $3000 per week and junior accountants earning $1000 per week. The contracts with companies to provide accounting services require at least 100 accountants, of whom at least 5 must be senior accountants. To make sure there is enough experience on the accounting team, the number of senior accountants should be at least a third of the number to junior accountants. The firm wants to keep the weekly wage bill below $150000. Formulate an LP to minimize the wage bill. \nformulation: \nVariables:\n x: senior accountants, y: junior accountants\nObjective Function:\nminimize\n3000.00 * x + 1000.00 * y\nConstraints:\n1.00 * x + 1.00 * y \u2265 100.00\n1.00 * x \u2265 5.00\n1.00 * x  \u2265 0.33 * y \n3000.00 * x + 1000.00 * y \u2264 150000.00. \n Give me the formulation of the below problem: {input_text} \nAssistant: Sure! \nformulation: "




        # labels = batch['label'].to(device)
        with no_ssl_verification():
            predicted_label= pipeline_llm(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer_llm.eos_token_id,
            max_length=4000,)
            response=predicted_label[0]['generated_text']
            if args.dataset=='svamp':
                idx0=response.find('Assistant:')
                idx1=response.find('Equation:',idx0)
                idx2=response.find('\n',idx1)
                idx3=response.find('=',idx1)
                if idx3==-1:
                    idx4=idx3
                elif idx2==-1:
                    idx4=idx2
                else:
                    idx4=max(idx3,idx2)
                response=response[idx1+9:idx4].replace('x','*')
                
                # response=response[:idx3]
                response=remove_letters(response)

            elif args.dataset in ['squad','adversarialQA']:
                idx0=response.find('Assistant:')
                idx1=response.find('is:',idx0)
                predicted_label=response[idx1+3:].replace('\n','').replace('"','').replace('.','').replace('answer:','').strip().lower()
            elif args.dataset=='nl4opt':
                idx0=response.find('Assistant:')
                idx1=response.find('\nformulation:',idx0)
                idx2=response.find('Please',idx1)
                predicted_label=response[idx1+14:idx2]
                pred_all+=predicted_label
                ref_all+=target_text  
                print('pred:',predicted_label,'label:',target_text) 
                


            if args.dataset=='svamp':
                print('pred:',response,'label:',target_text)
              
                predicted_label=eval_equation(response)
                target_text=eval_equation(target_text)
                if predicted_label == target_text:
                    total_correct += 1

            elif args.dataset in ['squad','adversarialQA']:
                target_text=target_text.lower().split('\n')
                print('pred:',predicted_label,'label:',target_text)
                
                
                

                for target_samp in target_text: 
                    if target_samp!='':
                        if target_samp in predicted_label or  predicted_label in target_samp:
                            total_correct += 1  
                            break
                              
            if args.dataset in ['squad','adversarialQA']:
                total_samples += 1
                accuracy = total_correct / total_samples
                print('Accuracy:',accuracy)
                 

    if args.dataset in ['squad','adversarialQA']:
        accuracy = total_correct / total_samples
        print('Accuracy:',accuracy)
    
    elif args.dataset in ['mtc','nl4opt']:
        results=rouge.compute(predictions=pred_all, references=ref_all)
        score=results["rougeL"].mid.fmeasure
        print(f'ROUGE-L: {score:.3f}')
