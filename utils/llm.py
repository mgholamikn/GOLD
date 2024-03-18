import os
import openai
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context
from .batching import *
import warnings
import contextlib

import requests
from urllib3.exceptions import InsecureRequestWarning
import numpy as np
import time
from .constants import topics
from .common import eval_equation, remove_letters

old_merge_environment_settings = requests.Session.merge_environment_settings

@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass


def llama2_anli(pipeline,tokenizer,num_samples,label,fb=[]):

        # prepare feedback content
        fb_content=""
        for ii in range(len(fb)):
            fb_content+=f"\n\n 1. \n\n Premise: {fb[ii]['sentence1']}\n\n Hypothesis:{fb[ii]['sentence2']} \n\n Label:{fb[ii]['label']}" 
        if label==2:
            details=f"with a 'contradiciton' label numbered from 1 to {num_samples}. 'contradiction' label means that the hypothesis contradicts a fact mentioned in the premise."
        elif label==1:
            details=f"with a 'neutral' label numbered from 1 to {num_samples}. 'neutral' label means that the premise neither entail nor contradict the hypothesis. Hypothesis usually provide some information not mentioned in the premise at all."
        else:
            details=f" with a 'entailment' label numbered from 1 to {num_samples}. 'entailment' label means that the premise entails the hypothesis."
            
        prompt=f"You are a helpful assistant. You only reply once as 'Assistant' do not pretend to be a 'User'. Take a deep breath and think step-by-step. Below are 3 examples of an natural language inference dataset. Samples include a 'hypothesis' and a 'premise'. The label of sample is: 1. 'entailment' if premise entail hypothesis, 2. 'neutral' if premise neither entail nor contradict hypothesis. 3.'contradiction' if premise contradict hypothesis \n\n 1. \n\n Premise: The Parma trolleybus system (Italian: 'Rete filoviaria di Parma' ) forms part of the public transport network of the city and 'comune' of Parma, in the region of Emilia-Romagna, northern Italy. In operation since 1953, the system presently comprises four urban routes. \n\n Hypothesis: The trolleybus system has over 2 urban routes \n\n Label: entailment \n\n 2. \n\n Premise: The Centralia Massacre was an incident during the American Civil War in which twenty-four unarmed Union soldiers were captured and executed at Centralia, Missouri on September 27, 1864 by the pro-Confederate guerrilla leader William T. Anderson. Future outlaw Jesse James was among the guerrillas. \n\n Hypothesis: Jesse James was a guerrilla in the Union army during the American Civil War. \n\n Label: contradiction \n\n 3. \n\n Premise: Alexandra Lendon Bastedo (9 March 1946 – 12 January 2014) was a British actress, best known for her role as secret agent Sharron Macready in the 1968 British espionage/science fiction adventure series 'The Champions'. She has been cited as a sex symbol of the 1960s and 1970s. Bastedo was a vegetarian and animal welfare advocate.\n\n Hypothesis:Sharron Macready was a popular character through the 1980 s. \n\n Label: neutral. \n\n The above are samples of natural language inference data. Give me {num_samples} samples of novel natural language inference data with {details}. Assistant:"

        # Send text to LLama2
        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4000,
        )
        response=sequences[0]['generated_text']
        idx0=response.find('Assistant:')
        response=response[idx0:]

        return response

def llama2_anli_val(num_samples,label,fb=[]):

        # prepare feedback content
        fb_content=""
        for ii in range(len(fb)):
            fb_content+=f"\n\n 1. \n\n Premise: {fb[ii]['sentence1']}\n\n Hypothesis:{fb[ii]['sentence2']} \n\n Label:{fb[ii]['label']}" 

        if label==2:
            details=f" with 'contradiciton' label that are significantly different from the above examples numbered from 1 to {num_samples}. Samples should be from a different domain, written from a different perspective, or written by a differnet charachter. 'contradiction' label means that hypothesis contradicts a fact mentioned in premise."
        elif label==1:
            details=f" with 'neutral' label that are significantly different from the above examples numbered from 1 to {num_samples}. Samples should be from a different domain, written from a different perspective, or written by a differnet charachter. 'neutral' label means that premise neither entail nor contradict hypothesis. Hypothesis usually provide some information not mentioned in the premise at all."
        else:
            details=f" with 'entailment' label that are significantly different from the above examples numbered from 1 to {num_samples}. Samples should be from a different domain, written from a different perspective, or written by a differnet charachter. 'entailment' label means premise entails hypothesis explicitly or pre logically."

        prompt=f"You are a helpful assistant. You only reply once as 'Assistant' do not pretend to be a 'User'. Take a deep breath and think step-by-step. Below are 3 examples of an natural language inference dataset. Samples include a 'hypothesis' and a 'premise'. The label of sample is: 1. 'entailment' if premise entail hypothesis, 2. 'neutral' if premise neither entail nor contradict hypothesis. 3.'contradiction' if premise contradict hypothesis \n\n 1. \n\n Premise: The Parma trolleybus system (Italian: 'Rete filoviaria di Parma' ) forms part of the public transport network of the city and 'comune' of Parma, in the region of Emilia-Romagna, northern Italy. In operation since 1953, the system presently comprises four urban routes. \n\n Hypothesis: The trolleybus system has over 2 urban routes \n\n Label: entailment \n\n 2. \n\n Premise: The Centralia Massacre was an incident during the American Civil War in which twenty-four unarmed Union soldiers were captured and executed at Centralia, Missouri on September 27, 1864 by the pro-Confederate guerrilla leader William T. Anderson. Future outlaw Jesse James was among the guerrillas. \n\n Hypothesis: Jesse James was a guerrilla in the Union army during the American Civil War. \n\n Label: contradiction \n\n 3. \n\n Premise: Alexandra Lendon Bastedo (9 March 1946 – 12 January 2014) was a British actress, best known for her role as secret agent Sharron Macready in the 1968 British espionage/science fiction adventure series 'The Champions'. She has been cited as a sex symbol of the 1960s and 1970s. Bastedo was a vegetarian and animal welfare advocate.\n\n Hypothesis:Sharron Macready was a popular character through the 1980 s. \n\n Label: neutral. \n\n The above are samples of natural language inference data. Give me {num_samples} samples of novel natural language inference data with {details}. Assistant:"

        # Send text to LLama2
        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4000,
        )
        response=sequences[0]['generated_text']
        idx0=response.find('Assistant:')
        response=response[idx0:]

        return response

 
def llama2_qnli(pipeline,tokenizer,num_samples,label,fb=[]):

        # prepare feedback content
        # fb_content=""
        
        fb_content="\n\n 1. \nSentence: He must do this by collecting the multiple Tears of Light; once all the Tears of Light are collected for one area, he restores that area's Light Spirit. \nQuestion: What does Link have to gather in order to complete each area? \nLabel: entailment \n\n 2. \nSentence:Prior to this time congressional parties were often relatively disorganized, so it was not always evident who functioned as the opposition floor leader. \nQuestion: Why was minority leader position created?. \nLabel: entailment \n\n 3. \nSentence:This view is shared by other researchers who argue that the ancestors of the American Indians were the first to separate from the great Asian population in the Middle Paleolithic. \nQuestion:Who have studies of the mtDNA of Turkic-speaking peoples shown they're closest to genetically? \nLabel: not_entailment"
        if fb!=[]:
            for ii in range(len(fb)):
                fb_content+=f"\n\n {str(ii+4)}. \nSentence: {fb[ii]['sentence']}\nQuestion:{fb[ii]['question']} \nLabel:{fb[ii]['label']}" 

        prompt1=f"You are a helpful 'Assistant'. You only reply once as 'Assistant'. Do not pretend to be a 'User'. Below are three samples of QNLI dataset. Samples include a 'Question' and a 'Sentence'. The label of sample is 1.'entailment' if the answer of the 'Question' is in the 'Sentence' and 2.'not_entailment' if the answer of the 'Question' is not in the 'Sentence'. {fb_content}  \n\n The above are three samples of QNLI data. \n Think step by step and give me a novel sample of QNLI data with not_entailment label (The Sentence does not include the answer of the Question). \n Assistant: Sure! here is a novel sample for you. The Question can 'not' be answered given the Sentence since the question is asking about some information 'not' mentioned in the Sentence at all. Sentence:"

        prompt0=f"You are a helpful 'Assistant'. You only reply once as 'Assistant'. Do not pretend to be a 'User'. Below are three samples of QNLI dataset. Samples include a 'Question' and a 'Sentence'. The label of sample is 1.'entailment' if the answer of the 'Question' is in the 'Sentence' and 2.'not_entailment' if the answer of the 'Question' is not in the 'Sentence'. {fb_content}  \n\n The above are three samples of QNLI data. \n Think step by step and give me a novel sample of QNLI data with entailment label (The Sentence includes the answer of the Question). \n Assistant: Sure! here is a novel sample for you. The Question can be answered given the Sentence since the question is asking about some facts directly mentioned in the Sentence. Sentence:"
        response_all=[]
                
        prompts=[prompt0,prompt1,]
        labels=["entailment","not_entailment"]
        idxs=np.random.randint(0,2,8)
        for ii in range(num_samples):
            idx=idxs[ii]
            prompt=prompts[idx]
            label=labels[idx]
            temp_data={}
            # Send text to LLama2
            sequences = pipeline(
                prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=4000,
            )
            response=sequences[0]['generated_text']
            idx0=response.find('Assistant:')
            idx1=response.find('Sentence:',idx0)
            idx2=response.find('Question:',idx1)
            idx3=response.find('?',idx2)
            temp_data["sentence"]=response[idx1+9:idx2]
            temp_data["question"]=response[idx2+9:idx3+1]
            temp_data["label"]=label
            response_all.append(temp_data)

        return response_all


 
def llama2_rte(pipeline,tokenizer,num_samples,label,fb=[],val=False):

        # prepare feedback content
        if val:
            val_comment='The following sample is significantly different from the above samples. The follwoing samples are from a different topics and from a different domain.'
        else:
            val_comment=''

        fb_content="\n\n 1. \nSentence1: No Weapons of Mass Destruction Found in Iraq Yet. \nSentence2: Weapons of Mass Destruction Found in Iraq. \nLabel: not_entailment \n\n 2. \nSentence1: A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI. \nSentence2: Pope Benedict XVI is the new leader of the Roman Catholic Church.  \nLabel: entailment \n\n 3. \nSentence1:Herceptin was already approved to treat the sickest breast cancer patients, and the company said, Monday, it will discuss with federal regulators the possibility of prescribing the drug for more breast cancer patients. \nSentence2:Herceptin can be used to treat breast cancer. \nLabel: entailment \n\n"

        if fb!=[]:
            for ii in range(len(fb)):
                fb_content+=f"\n\n {str(ii+4)}. \nSentence1: {fb[ii]['sentence1']}\nSentence2:{fb[ii]['sentence2']} \nLabel:{fb[ii]['label']}" 

        prompt1=f"You are a helpful 'Assistant'. You only reply once as 'Assistant'. Do not pretend to be a 'User'. Below are three samples of RTE dataset. Samples include a 'Sentence1' and a 'Sentence2'. The label of sample is 'entailment' if the answer of the 'Sentence1' entails 'Sentence2' and 'not_entailment' if 'Sentence1' does not entail 'Sentence2'. {fb_content} The above are three samples of RTE data. \n Think step by step and give me a novel sample of RTE data with not_entailment label (The Sentence1 does not enatil Sentence2). \n Assistant: Sure! here is a novel sample for you. The Sentence1 is long and comprehensive following samples that you provided and Sentence1 does not entail Sentence2 which means that the Sentence2 either contradicts Sentence1 or provides some information not mentioned in Sentence1. {val_comment} Sentence1:"

        prompt0=f"You are a helpful 'Assistant'. You only reply once as 'Assistant'. Do not pretend to be a 'User'. Below are three samples of RTE dataset. Samples include a 'Sentence1' and a 'Sentence2'. The label of sample is 'entailment' if the answer of the 'Sentence1' entails 'Sentence2' and 'not_entailment' if 'Sentence1' does not entail 'Sentence2'. {fb_content} The above are three samples of RTE data. \n Think step by step and give me a novel sample of RTE data with entailment label (The Sentence1 enatils Sentence2). \n Assistant: Sure! here is a novel sample for you. The Sentence1 is long and comprehensive following samples that you provided and Sentence1 entails Sentence2 which means that the Sentence2 can be inferred from Sentence1. {val_comment} Sentence1:"
        response_all=[]
                
        prompts=[prompt0,prompt1,]
        labels=["entailment","not_entailment"]
        idxs=np.random.randint(0,2,8)
        for ii in range(num_samples):
            idx=idxs[ii]
            prompt=prompts[idx]
            label=labels[idx]
            temp_data={}
            # Send text to LLama2
            sequences = pipeline(
                prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=4000,
            )
            response=sequences[0]['generated_text']
            idx0=response.find('Assistant:')
            idx1=response.find('Sentence1:',idx0)
            idx2=response.find('Sentence2:',idx1)
            idx3=response.find('Label',idx2)
            temp_data["sentence1"]=response[idx1+10:idx2].replace('\n','').strip()
            temp_data["sentence2"]=response[idx2+10:idx3].replace('\n','').strip()
            temp_data["label"]=label
            if len(temp_data["sentence2"])<400 and len(temp_data["sentence2"])>10 and len(temp_data["sentence1"])<400 and len(temp_data["sentence1"])>10:
                response_all.append(temp_data)

        return response_all

 
def llama2_qnli_val(pipeline,tokenizer,num_samples,label,fb=[]):

        # prepare feedback content
        # fb_content=""
        # if fb==[]:
        fb_content="\n\n 1. \nSentence: He must do this by collecting the multiple Tears of Light; once all the Tears of Light are collected for one area, he restores that area's Light Spirit. \nQuestion: What does Link have to gather in order to complete each area? \nLabel: entailment \n\n 2. \nSentence:Prior to this time congressional parties were often relatively disorganized, so it was not always evident who functioned as the opposition floor leader. \nQuestion: Why was minority leader position created?. \nLabel: entailment \n\n 3. \nSentence:This view is shared by other researchers who argue that the ancestors of the American Indians were the first to separate from the great Asian population in the Middle Paleolithic. \nQuestion:Who have studies of the mtDNA of Turkic-speaking peoples shown they're closest to genetically? \nLabel: not_entailment"
        if fb!=[]:
            for ii in range(len(fb)):
                fb_content+=f"\n\n {str(ii+4)}. \nSentence: {fb[ii]['sentence']}\nQuestion:{fb[ii]['question']} \nLabel:{fb[ii]['label']}" 

        prompt1=f"You are a helpful 'Assistant'. You only reply once as 'Assistant'. Do not pretend to be a 'User'. Below are three samples of QNLI dataset. Samples include a 'Question' and a 'Sentence'. The label of sample is 1.'entailment' if the answer of the 'Question' is in the 'Sentence' and 2.'not_entailment' if the answer of the 'Question' is not in the 'Sentence'. {fb_content}  \n\n The above are three samples of QNLI data. \n Think step by step and give me a novel sample of QNLI data with not_entailment label (The Sentence does not include the answer of the Question). \n Assistant: Sure! here is a novel sample for you. The Question can 'not' be answered given the Sentence since the question is asking about some information 'not' mentioned in the Sentence at all. The following sample is significantly different from the above samples. Sentence:"

        prompt0=f"You are a helpful 'Assistant'. You only reply once as 'Assistant'. Do not pretend to be a 'User'. Below are three samples of QNLI dataset. Samples include a 'Question' and a 'Sentence'. The label of sample is 1.'entailment' if the answer of the 'Question' is in the 'Sentence' and 2.'not_entailment' if the answer of the 'Question' is not in the 'Sentence'. {fb_content}  \n\n The above are three samples of QNLI data. \n Think step by step and give me a novel sample of QNLI data with entailment label (The Sentence includes the answer of the Question). \n Assistant: Sure! here is a novel sample for you. The Question can be answered given the Sentence since the question is asking about some facts directly mentioned in the Sentence. The following sample is significantly different from the above samples. Sentence:"
        response_all=[]
        for ii in range(num_samples//2):
            temp_data={}
            # Send text to LLama2
            sequences = pipeline(
                prompt0,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=4000,
            )
            response=sequences[0]['generated_text']
            idx0=response.find('Assistant:')
            idx1=response.find('Sentence:',idx0)
            idx2=response.find('Question:',idx1)
            idx3=response.find('?',idx2)
            temp_data["sentence"]=response[idx1+9:idx2]
            temp_data["question"]=response[idx2+9:idx3+1]
            temp_data["label"]="entailment"
            response_all.append(temp_data)
        for ii in range(num_samples//2):
            temp_data={}
            # Send text to LLama2
            sequences = pipeline(
                prompt1,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=4000,
            )
            response=sequences[0]['generated_text']
            idx0=response.find('Assistant:')
            idx1=response.find('Sentence:',idx0)
            idx2=response.find('Question:',idx1)
            idx3=response.find('?',idx2)
            temp_data["sentence"]=response[idx1+9:idx2]
            temp_data["question"]=response[idx2+9:idx3+1]
            temp_data["label"]="not_entailment"
            response_all.append(temp_data)

        return response_all


def llama2_anli(pipeline,tokenizer,num_samples,label,fb=[]):

        # prepare feedback content
        # fb_content=""
        
        fb_content="\n\n 1. \nSentence: He must do this by collecting the multiple Tears of Light; once all the Tears of Light are collected for one area, he restores that area's Light Spirit. \nQuestion: What does Link have to gather in order to complete each area? \nLabel: entailment \n\n 2. \nSentence:Prior to this time congressional parties were often relatively disorganized, so it was not always evident who functioned as the opposition floor leader. \nQuestion: Why was minority leader position created?. \nLabel: entailment \n\n 3. \nSentence:This view is shared by other researchers who argue that the ancestors of the American Indians were the first to separate from the great Asian population in the Middle Paleolithic. \nQuestion:Who have studies of the mtDNA of Turkic-speaking peoples shown they're closest to genetically? \nLabel: not_entailment"
        if fb!=[]:
            for ii in range(len(fb)):
                fb_content+=f"\n\n {str(ii+4)}. \nSentence: {fb[ii]['sentence']}\nQuestion:{fb[ii]['question']} \nLabel:{fb[ii]['label']}" 

        prompt1=f"You are a helpful 'Assistant'. You only reply once as 'Assistant'. Do not pretend to be a 'User'. Below are three samples of QNLI dataset. Samples include a 'Question' and a 'Sentence'. The label of sample is 1.'entailment' if the answer of the 'Question' is in the 'Sentence' and 2.'not_entailment' if the answer of the 'Question' is not in the 'Sentence'. {fb_content}  \n\n The above are three samples of QNLI data. \n Think step by step and give me a novel sample of QNLI data with not_entailment label (The Sentence does not include the answer of the Question). \n Assistant: Sure! here is a novel sample for you. The Question can 'not' be answered given the Sentence since the question is asking about some information 'not' mentioned in the Sentence at all. Sentence:"

        prompt0=f"You are a helpful 'Assistant'. You only reply once as 'Assistant'. Do not pretend to be a 'User'. Below are three samples of QNLI dataset. Samples include a 'Question' and a 'Sentence'. The label of sample is 1.'entailment' if the answer of the 'Question' is in the 'Sentence' and 2.'not_entailment' if the answer of the 'Question' is not in the 'Sentence'. {fb_content}  \n\n The above are three samples of QNLI data. \n Think step by step and give me a novel sample of QNLI data with entailment label (The Sentence includes the answer of the Question). \n Assistant: Sure! here is a novel sample for you. The Question can be answered given the Sentence since the question is asking about some facts directly mentioned in the Sentence. Sentence:"
        response_all=[]
        for ii in range(num_samples//2):
            temp_data={}
            # Send text to LLama2
            sequences = pipeline(
                prompt0,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=4000,
            )
            response=sequences[0]['generated_text']
            idx0=response.find('Assistant:')
            idx1=response.find('Sentence:',idx0)
            idx2=response.find('Question:',idx1)
            idx3=response.find('?',idx2)
            temp_data["sentence"]=response[idx1+9:idx2]
            temp_data["question"]=response[idx2+9:idx3+1]
            temp_data["label"]="entailment"
            response_all.append(temp_data)
        for ii in range(num_samples//2):
            temp_data={}
            # Send text to LLama2
            sequences = pipeline(
                prompt1,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=4000,
            )
            response=sequences[0]['generated_text']
            idx0=response.find('Assistant:')
            idx1=response.find('Sentence:',idx0)
            idx2=response.find('Question:',idx1)
            idx3=response.find('?',idx2)
            temp_data["sentence"]=response[idx1+9:idx2]
            temp_data["question"]=response[idx2+9:idx3+1]
            temp_data["label"]="not_entailment"
            response_all.append(temp_data)

        return response_all

 
def llama2_anli_val(pipeline,tokenizer,num_samples,label,fb=[]):

        # prepare feedback content
        # fb_content=""
        # if fb==[]:
        fb_content="\n\n 1. \nSentence: He must do this by collecting the multiple Tears of Light; once all the Tears of Light are collected for one area, he restores that area's Light Spirit. \nQuestion: What does Link have to gather in order to complete each area? \nLabel: entailment \n\n 2. \nSentence:Prior to this time congressional parties were often relatively disorganized, so it was not always evident who functioned as the opposition floor leader. \nQuestion: Why was minority leader position created?. \nLabel: entailment \n\n 3. \nSentence:This view is shared by other researchers who argue that the ancestors of the American Indians were the first to separate from the great Asian population in the Middle Paleolithic. \nQuestion:Who have studies of the mtDNA of Turkic-speaking peoples shown they're closest to genetically? \nLabel: not_entailment"
        if fb!=[]:
            for ii in range(4,len(fb)):
                fb_content+=f"\n\n {str(ii)}. \nSentence: {fb[ii]['sentence']}\nQuestion:{fb[ii]['question']} \nLabel:{fb[ii]['label']}" 

        prompt1=f"You are a helpful 'Assistant'. You only reply once as 'Assistant'. Do not pretend to be a 'User'. Below are three samples of QNLI dataset. Samples include a 'Question' and a 'Sentence'. The label of sample is 1.'entailment' if the answer of the 'Question' is in the 'Sentence' and 2.'not_entailment' if the answer of the 'Question' is not in the 'Sentence'. {fb_content}  \n\n The above are three samples of QNLI data. \n Think step by step and give me a novel sample of QNLI data with not_entailment label (The Sentence does not include the answer of the Question). \n Assistant: Sure! here is a novel sample for you. The Question can 'not' be answered given the Sentence since the question is asking about some information 'not' mentioned in the Sentence at all. The following sample is significantly different from the above samples. Sentence:"

        prompt0=f"You are a helpful 'Assistant'. You only reply once as 'Assistant'. Do not pretend to be a 'User'. Below are three samples of QNLI dataset. Samples include a 'Question' and a 'Sentence'. The label of sample is 1.'entailment' if the answer of the 'Question' is in the 'Sentence' and 2.'not_entailment' if the answer of the 'Question' is not in the 'Sentence'. {fb_content}  \n\n The above are three samples of QNLI data. \n Think step by step and give me a novel sample of QNLI data with entailment label (The Sentence includes the answer of the Question). \n Assistant: Sure! here is a novel sample for you. The Question can be answered given the Sentence since the question is asking about some facts directly mentioned in the Sentence. The following sample is significantly different from the above samples. Sentence:"
        response_all=[]
        for ii in range(num_samples//2):
            temp_data={}
            # Send text to LLama2
            sequences = pipeline(
                prompt0,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=4000,
            )
            response=sequences[0]['generated_text']
            idx0=response.find('Assistant:')
            idx1=response.find('Sentence:',idx0)
            idx2=response.find('Question:',idx1)
            idx3=response.find('?',idx2)
            temp_data["sentence"]=response[idx1+9:idx2]
            temp_data["question"]=response[idx2+9:idx3+1]
            temp_data["label"]="entailment"
            response_all.append(temp_data)
        for ii in range(num_samples//2):
            temp_data={}
            # Send text to LLama2
            sequences = pipeline(
                prompt1,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=4000,
            )
            response=sequences[0]['generated_text']
            idx0=response.find('Assistant:')
            idx1=response.find('Sentence:',idx0)
            idx2=response.find('Question:',idx1)
            idx3=response.find('?',idx2)
            temp_data["sentence"]=response[idx1+9:idx2]
            temp_data["question"]=response[idx2+9:idx3+1]
            temp_data["label"]="not_entailment"
            response_all.append(temp_data)

        return response_all


def llama2_squad(pipeline,tokenizer,num_samples,label,fb=[],val=False):

        # prepare feedback content
        if val:
            val_comment='The following sample is significantly different from the above samples. It is from a new topic'
        else:
            val_comment=''

        fb_content="\n\n The above are samples of SQUAD data. It has a 'context' which is a paragraph from wikipedia, a 'question' from the paragraph and a short 'answer' for the question. 'answers' are directly from the 'context'. \n\n 1. context: Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend 'Venite Ad Me Omnes'. Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary. question:To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France? answer:Saint Bernadette Soubirous \n\n 2. context: As at most other universities, Notre Dame's students run a number of news media outlets. The nine student-run outlets include three newspapers, both a radio and television station, and several magazines and journals. Begun as a one-page journal in September 1876, the Scholastic magazine is issued twice monthly and claims to be the oldest continuous collegiate publication in the United States. The other magazine, The Juggler, is released twice a year and focuses on student literature and artwork. The Dome yearbook is published annually. The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news, and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent publication and does not have a faculty advisor or any editorial oversight from the University. In 1987, when some students believed that The Observer began to show a conservative bias, a liberal newspaper, Common Sense was published. Likewise, in 2003, when other students believed that the paper showed a liberal bias, the conservative paper Irish Rover went into production. Neither paper is published as often as The Observer; however, all three are distributed to all students. Finally, in Spring 2008 an undergraduate journal for political science research, Beyond Politics, made its debut. question: In what year did the student paper Common Sense begin publication at Notre Dame? answer: 1987 \n\n 3. context: The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study – aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering – with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively. question: How many departments are within the Stinson-Remick Hall of Engineering? answer: fiven"

        if fb!=[]:
            for ii in range(len(fb)):
                fb_content+=f"\n\n {str(ii+3)}. context: {fb[ii]['context']}. question:{fb[ii]['question']}. answer:{fb[ii]['answer']}" 

        prompt=f"{fb_content} Generate 1 novel sample from a new topic. \n Assistant: Sure! here is a novel sample for you. It has a 'context' and a 'question'. The 'answer' of the question is in the 'context' and the 'answer' is as short as 2 to 4 words from the context. {val_comment}. context:"

        response_all=[]
                

        while len(response_all)<8:
            temp_data={}
            # Send text to LLama2
            sequences = pipeline(
                prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=4000,
            )
            response=sequences[0]['generated_text']
            idx0=response.find('Assistant:')
            idx1=response.find('context:',idx0)
            idx2=response.find('question:',idx1)
            idx3=response.find('answer:',idx2)
            temp_data["context"]=response[idx1+8:idx2].replace('\n','').strip()
            temp_data["question"]=response[idx2+9:idx3].replace('\n','').strip()
            temp_data["answer"]=response[idx3+7:].replace('\n','').strip()
            if len(temp_data["context"])<400 and len(temp_data["context"])>10 and len(temp_data["question"])<400 and len(temp_data["question"])>10 and temp_data["answer"].lower() in temp_data["context"].lower():
                response_all.append(temp_data)

        return response_all




def llama2_svamp(pipeline,tokenizer,num_samples,label,fb=[],val=False):

        # prepare feedback content
        if val:
            val_comment='The Body of the sample is significantly different from the above samples and it is from a different topic and domain.'
        else:
            val_comment=''

        fb_content="You are a helpful assistant. You only reply once as 'Assistant' do not pretend to be a 'User'. Take a deep breath and think step-by-step. Below are 3 examples of SVAMP dataset. Samples include a 'Body' which is explains a simple math problem and a 'Quesiton' which ask a question from the 'Body'. The label is an 'Equation' that solves the 'Question'. \n\n 1. Body:There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups. Question: How big is each group of bananas? Equation:( 290.0 / 2.0 ) Answer: 145 \n\n 2. Body: Marco and his dad went strawberry picking. Marco's dad's strawberries weighed 11 pounds. If together their strawberries weighed 30 pounds. Question: How much did Marco's strawberries weigh? Equation: ( 30.0 - 11.0 ) Answer: 19 \n\n 3. Body: Edward spent $ 6 to buy 2 books each book costing him the same amount of money. Now he has $ 12. Question: How much did each book cost? Equation: ( 6.0 / 2.0 ) Answer: 3"

        if fb!=[]:
            for ii in range(len(fb)):
                fb_content+=f"\n\n {str(ii+4)}. Body: {fb[ii]['Body']} Question:{fb[ii]['Question']} Equation:{fb[ii]['Equation']} Answer:{fb[ii]['Answer']}" 

        prompt=f"{fb_content} \n\n Give me a novel sample of SVAMP data. Assistant: Sure! here is a novel sample. {val_comment}. Body:"

        response_all=[]
                

        while len(response_all)<8:
            temp_data={}
            # Send text to LLama2
            sequences = pipeline(
                prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=4000,
            )
            response=sequences[0]['generated_text']
            idx0=response.find('Assistant:')
            idx1=response.find('Body:',idx0)
            idx2=response.find('Question:',idx1)
            idx3=response.find('Equation:',idx2)
            idx4=response.find('Answer:',idx3)

            temp_data["Body"]=response[idx1+5:idx2].replace('\n','').strip()
            temp_data["Question"]=response[idx2+9:idx3].replace('\n','').strip()
            temp_data["Equation"]=response[idx3+9:idx4].replace('\n','').replace('x','*').replace(',','').strip()
            # temp_data["Equation"]=remove_letters(response[idx3+9:idx4].replace('\n','').replace('x','*').replace(',','').strip())

            temp_data["Answer"]=response[idx4+7:]
            if len(temp_data["Body"])<400 and len(temp_data["Body"])>10 and len(temp_data["Question"])<400 and len(temp_data["Question"])>10 and eval_equation(temp_data["Equation"]) is not np.nan and len(temp_data["Equation"])<400 and len(temp_data["Equation"])>10:
                response_all.append(temp_data)

        return response_all




def llama2_nl4opt(pipeline,tokenizer,num_samples,label,fb=[],val=False):

        # prepare feedback content
        if val:
            val_comment='The following sample is significantly different from the above samples and it is from a different topic and domain.'
        else:
            val_comment=''
        fb_content=''
        if fb!=[]:
            for ii in range(len(fb)):
                fb_content+=f"\n\n {str(ii)}. \nproblem: {fb[ii]['problem']}, \nformulation:{fb[ii]['formulation']}" 

        fb_content+="You are a helpful assistant. You only reply once as 'Assistant' do not pretend to be a 'User'. The above are examples of NL4OPT dataseta and below are three samples of 'NL4OPT' dataset. The samples have a 'problem' which is a linear optimization problem and a 'formulation' which is the formulation of the optimization problem. \n\n1. \nproblem: A hotel employs cleaners and receptionists. Cleaners earn $500 per week and receptionists earn $350 per week. The hotel requires a minimum of 100 workers of whom at least 20 must be receptionists. To keep the hotel clean and running smoothly, the number of receptionists should be at least a third of the number of cleaners. The hotel wants to keep the weekly wage bill below $30000. Formulate a LP to minimize the wage bill. \nformulation: \nVariables:\n x: sled dogs, y: trucks\nObjective Function:\nmaximize\n100.00 * x + 300.00 * y\nConstraints:\n50.00 * x + 100.00 * y \u2264 1000.00\n1.00 * x  \u2264 1.00 * y \n\n2. \nproblem: An office supply company makes two types of printers: color printers and black and white printers. Different sections of the factory with different teams produce each printer. The color printer team can produce at most 20 color printers per day while the black and white printer team can produce at most 30 black and white printers per day. Both teams require use of the same paper tray installing machine and this machine can make at most 35 printers of either type each day. Color printers generate a profit of $200 per printer while black and white printers generate a profit of $70 per printer. How many of each printer should be made to maximize the company's profit? \nformulation: \nVariables:\n x: color printers, y: black and white printers\nObjective Function:\nmaximize\n200.00 * x + 70.00 * y\nConstraints:\n1.00 * x \u2264 20.00\n1.00 * y \u2264 30.00\n1.00 * x + 1.00 * y \u2264 35.00 \n\n3. \nproblem: An accounting firm has senior accountants earning $3000 per week and junior accountants earning $1000 per week. The contracts with companies to provide accounting services require at least 100 accountants, of whom at least 5 must be senior accountants. To make sure there is enough experience on the accounting team, the number of senior accountants should be at least a third of the number to junior accountants. The firm wants to keep the weekly wage bill below $150000. Formulate an LP to minimize the wage bill. \nformulation: \nVariables:\n x: senior accountants, y: junior accountants\nObjective Function:\nminimize\n3000.00 * x + 1000.00 * y\nConstraints:\n1.00 * x + 1.00 * y \u2265 100.00\n1.00 * x \u2265 5.00\n1.00 * x  \u2265 0.33 * y \n3000.00 * x + 1000.00 * y \u2264 150000.00."

        prompt=f"{fb_content} \nGive me a novel sample of NL4OPT data. \nAssistant: Sure! the following is a novel sample of NL4OPT dataset. I have followed the format of above samples in generating the 'formulation'. {val_comment}. \nproblem:"

        response_all=[]
                

        while len(response_all)<8:
            temp_data={}
            # Send text to LLama2
            sequences = pipeline(
                prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=4000,
            )
            
            response=sequences[0]['generated_text']
         
            idx0=response.find('\nAssistant:')
            idx1=response.find('problem:',idx0)
            idx2=response.find('Variables:',idx1)
            idx3=response.find('\nPlease',idx2)
            idx4=response.find('\nI hope',idx2)
            idx5=response.find('\nGive',idx2)
            idx6=response.find('\nDo you',idx2)
            idx7=response.find('\nNote',idx2)
            idx8=max([idx3,idx4,idx5,idx6,idx7])
            temp_data["problem"]=response[idx1+8:idx2].replace('\nformulation:','')
            temp_data["formulation"]=response[idx2:idx8].replace('\n\n\n','')
            print('&&&&&&&',temp_data)
            if len(temp_data["problem"])<600 and len(temp_data["problem"])>10 and len(temp_data["formulation"])<400 and len(temp_data["formulation"])>10:
                response_all.append(temp_data)

        return response_all
