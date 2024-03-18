import numpy as np
import random
def batching(text1,text2,num_entailment):
    # this function return batchs of data for entailment prediciton task, given chat-gpt outputs
    # input (text): 1. ... 10. Premise: The book was a bestseller. Hypothesis: The book was never published. Label: No_Entailment 11. ...
    # output (batch list): batch[10]: {'Premise': ' The students studied hard for the exam.', 'Hypothesis': ' The students failed the exam.', 'Label': ' No_Entailmen'}
    batch_dize=16
    
    batch=[]
    labels=[]

    
    # add entailment sample to batch
    index=np.zeros(num_entailment)
    for ii in range(num_entailment):
        index[ii]=int(text1.find(str(ii+1)+"."))

    for ii in range(num_entailment):
        labels.append(0)
        if ii==num_entailment-1:
            batch.append(text1[int(index[ii]):-1])
        else:
            batch.append(text1[int(index[ii]):int(index[ii+1])])

    # add not_entailment sample to batch
    index=np.zeros(batch_dize-num_entailment)
    for ii in range(batch_dize-num_entailment):
        labels.append(1)
        index[ii]=int(text2.find(str(ii+1)+"."))

    for ii in range(batch_dize-num_entailment):
        if ii==batch_dize-num_entailment-1:
            batch.append(text2[int(index[ii]):-1])
        else:
            batch.append(text2[int(index[ii]):int(index[ii+1])])

    
    new_batch=[]
    count=0
    for example in batch:
        data={}
        index_1=example.find("Premise:")
        index_2=example.find("\n",index_1)
        index_3=example.find('Hypothesis:')
        index_4=example.find("\n",index_3)
        # index_5=example.find("Label:")
        data["sentence1"]=example[12:index_2]
        data["sentence2"]=example[index_3+11:]
        # if "No" in example[index_5:]:
        #     data["label"]=1
        # else:
        #     data["label"]=0
        data["label"]=labels[count]

        new_batch.append(data)
        count+=1
    
    
    return new_batch

def batching_anli(text1,text2,text3,num_text1,num_text2,num_text3):
    # this function return batchs of data for entailment prediciton task, given chat-gpt outputs
    # input (text): 1. ... 10. Premise: The book was a bestseller. Hypothesis: The book was never published. Label: No_Entailment 11. ...
    # output (batch list): batch[10]: {'Premise': ' The students studied hard for the exam.', 'Hypothesis': ' The students failed the exam.', 'Label': ' No_Entailmen'}
    batch_dize=16
    
    batch=[]
    labels=[]

    
    # add entailment sample to batch
    for ii in range(num_text1):
        index1=int(text1.find(str(ii+1)+"."))
        index2=int(text1.find(str(ii+2)+"."))
        labels.append('entailment')
        if ii==num_text1-1:
            #to remove extra texts at the end of response
            index3=int(text3.find("\n",index2))
            batch.append(text1[int(index1):index3])
        else:
            batch.append(text1[int(index1):int(index2)])

    # add neutral sample to batch
    for ii in range(num_text2):
        index1=int(text2.find(str(ii+1)+"."))
        index2=int(text2.find(str(ii+2)+"."))
        labels.append('neutral')
        if ii==num_text2-1:
            #to remove extra texts at the end of response
            index3=int(text3.find("\n",index2))
            batch.append(text2[int(index1):index3])
        else:
            batch.append(text2[int(index1):int(index2)])

    # add contradiction sample to batch
    for ii in range(num_text3):
        index1=int(text3.find(str(ii+1)+"."))
        index2=int(text3.find(str(ii+2)+"."))
        
        labels.append('contradiction')
        if ii==num_text3-1:
            #to remove extra texts at the end of response
            index3=int(text3.find("\n",index2))
            batch.append(text3[int(index1):index3])
        else:
            batch.append(text3[int(index1):int(index2)])
    
    new_batch=[]
    count=0
    for example in batch:
        data={}
        index_1=example.find("Premise:")
        index_2=example.find("\n",index_1)
        index_3=example.find('Hypothesis:')
        index_4=example.find("Label")
        # index_5=example.find("Label:")
        data["sentence1"]=example[12:index_2]
        data["sentence2"]=example[index_3+11:index_4-2]
        # if "No" in example[index_5:]:
        #     data["label"]=1
        # else:
        #     data["label"]=0
        data["label"]=labels[count]

        new_batch.append(data)
        count+=1
    
    
    return new_batch

def batching_anli_2(text1,num_text1,):
    # this function return batchs of data for entailment prediciton task, given chat-gpt outputs
    # input (text): 1. ... 10. Premise: The book was a bestseller. Hypothesis: The book was never published. Label: No_Entailment 11. ...
    # output (batch list): batch[10]: {'Premise': ' The students studied hard for the exam.', 'Hypothesis': ' The students failed the exam.', 'Label': ' No_Entailmen'}
    batch_dize=16
    
    batch=[]
    labels=[]

    
    # add entailment sample to batch
    for ii in range(num_text1):
        index1=int(text1.find(str(ii+1)+"."))
        index2=int(text1.find(str(ii+2)+"."))
        labels.append('entailment')
        if ii==num_text1-1:
            batch.append(text1[int(index1):-1])
        else:
            batch.append(text1[int(index1):int(index2)])
    
    new_batch=[]
    count=0
    for example in batch:
        data={}
        index_1=example.find("Premise:")
        index_2=example.find("\n",index_1)
        index_3=example.find('Hypothesis:')
        index_4=example.find("Label")
        # index_5=example.find("Label:")
        data["sentence1"]=example[12:index_2]
        data["sentence2"]=example[index_3+11:index_4-2]
        # if "No" in example[index_5:]:
        #     data["label"]=1
        # else:
        #     data["label"]=0
        data["label"]=labels[count]

        new_batch.append(data)
        count+=1
    
    
    return new_batch

def new_batching(text1,text2,num_entailment):
    # this function gives return batchs of data for entailment prediciton task, given chat-gpt outputs
    # input (text): 1. ... 10. Premise: The book was a bestseller. Hypothesis: The book was never published. Label: No_Entailment 11. ...
    # output (batch list): batch[10]: {'Premise': ' The students studied hard for the exam.', 'Hypothesis': ' The students failed the exam.', 'Label': ' No_Entailmen'}
    batch_dize=16
    batch=[]
    index_0=0
    for ii in range(num_entailment):
        data={}
        index_1=text1.find("Premise:",index_0)
        index_2=text1.find("\n",index_1)
        index_3=text1.find('Hypothesis:',index_2)
        index_4=text1.find("\n",index_3) 
        index_0=text1.find("Premise:",index_3)
        # index_5=example.find("Label:")
        data["sentence1"]=text1[12:index_2]
        if ii==num_entailment-1:
            data["sentence2"]=text1[index_3+11:]
        else:
            data["sentence2"]=text1[index_3+11:index_4]

        data["label"]=0

        batch.append(data)

    index_0=0    
    for ii in range(batch_dize-num_entailment):
        data={}
        index_1=text1.find("Premise:",index_0)
        index_2=text1.find("\n",index_1)
        index_3=text1.find('Hypothesis:',index_2)
        index_4=text1.find("\n",index_3) 
        index_0=text1.find("Premise:",index_3)
        
        # index_5=example.find("Label:")
        data["sentence1"]=text1[12:index_2]
        if ii==batch_dize-num_entailment-1:
            data["sentence2"]=text1[index_3+11:]
        else:
            data["sentence2"]=text1[index_3+11:index_4]

        data["label"]=1

        batch.append(data)    
    
    return batch

def batching_nl4opt(text):
    
    # this function return batchs of data for nl4opt task, given chat-gpt outputs
    # input (text): Problem: A farmer wants to grow crops of cherries, apples, and oranges in his farm. He has an area of 20 acres available for planting these three crops, and he wants to maximize his profit. The profit per acre of cherry, apple, and orange crops is $500, $700, and $900 respectively. Moreover, he wants to plant at least 5 acres of cherries and at least 6 acres of apples. Determine the optimal acres of each crop that the farmer should plant to maximize his profit.\n\nVariables:\nx = acres of cherry crop to be planted\ny = acres of apple crop to be planted \nz = acres of orange crop to be planted\n\nObjective: \nMaximize \n 500x + 700y + 900z\n\nConstraints: \nx + y + z \u2264 20 \nx \u2265 5 \ny \u2265 6
    # output (batch list): {'Problem': ' ', 'Formulation': ' '}
    data={}
    idx1= text.find("Problem:")
    idx2= text.find("Variables:")
    data['Problem']=text[7:idx2]
    data['Formulation']=text[idx2:]
    
    
    return data

def batching_squad(text,num_text,):
    # this function return batchs of data for entailment prediciton task, given chat-gpt outputs
    # input (text): 1. ... 10. Premise: The book was a bestseller. Hypothesis: The book was never published. Label: No_Entailment 11. ...
    # output (batch list): batch[10]: {'Premise': ' The students studied hard for the exam.', 'Hypothesis': ' The students failed the exam.', 'Label': ' No_Entailmen'}
    
    batch=[]
    
    # add entailment sample to batch
    for ii in range(num_text):
        index1=int(text.find(str(ii+1)+"."))
        index2=int(text.find(str(ii+2)+"."))
        if ii==num_text-1:
            batch.append(text[int(index1):-1])
        else:
            batch.append(text[int(index1):int(index2)])
    
    new_batch=[]
   
    for example in batch:
        data={}
        
        index_1=example.lower().find("context:")
        index_2=example.lower().find("\n",index_1)
        index_3=example.lower().find('question:')
        index_4=example.lower().find("answer:")
        index_5=example.lower().find("\n",index_4)
        # index_5=example.find("Label:")
        data["context"]=example[index_1+8:index_3].strip()
        data["question"]=example[index_3+9:index_4-2].strip()
        data["answer"]=example[index_4+7:index_5].strip()

        new_batch.append(data)
    
    
    return new_batch