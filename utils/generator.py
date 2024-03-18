from utils.chatgpt import chatgpt_rte_v2, chatgpt_qnli
from utils.batching import *

def generator(dataset='rte'):

    if dataset=='rte':
        # call chat-gpt to generate data
        num_entailment=np.random.randint(6,10)
        details=True if np.random.rand()>0.4 else False 
        response1=chatgpt_rte_v2(num_entailment,entailment=True,details=details)
        response2=chatgpt_rte_v2(16-num_entailment,entailment=False,details=details)
        new_batch=batching(response1,response2,num_entailment)
        return new_batch
    
    if dataset=='qnli':
        # call chat-gpt to generate data
        num_entailment=np.random.randint(6,10)
        details=True if np.random.rand()>0.4 else False 
        response1=chatgpt_qnli(num_entailment,entailment=True,details=details)
        response2=chatgpt_qnli(16-num_entailment,entailment=False,details=details)
        new_batch=batching(response1,response2,num_entailment)
        return new_batch
