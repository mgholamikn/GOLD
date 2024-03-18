
import numpy as np
import re
import torch

def algeb2canon(text):
    
    idx2=text.find('Objective')
    idx1=text.find('Variables')
    idx3=text.find('Constraints')
    idx4=text.find('Formatting')
    text=text.replace("x1","x").replace("x2","y").replace("x3","z").replace("=","≤")
    
    text1=text[:idx1]
    text2=text[idx1:idx2]
    text3=text[idx2:idx3]
    text4=text[idx3:idx4]
    # print(text1,'\n')
    # print(text2,'\n')
    # print(text3,)
    # print(text4,)
    if ' z' in text4:
        max_col=4
    else:
        max_col=3
    text3=text3.split('\n')
    text4=text4.split('\n')[1:]

    if len(text3)>1:
        text3=text3[1:]
    for ii in range(len(text4)):
        if ':' in text4[ii]:
            idx=min(text4[ii].find(':')+1,len(text4[ii])-1)
            text4[ii]=text4[ii][idx:]
    text4=[x for x in text4 if x.strip()]
    for ii in range(len(text3)):
        if ':' in text3[ii]:
            idx=min(text3[ii].find(':')+1,len(text3[ii])-1)
            text3[ii]=text3[ii][idx:]
    text3=[x for x in text3 if x.strip()]  
    print(text3,)
    print(text4,)
   
    # idx_list=[]
    # idx=0
    # for term in text4:
    #     if '≥' in term:
    #         idx_list.append(idx)
    #     idx+=1
    # for idx in idx_list:
    #     print(text4[idx+1)

    a1=[]
    obj_dir=[1]
    for line in text3:
        line=line.replace("-","+ -")
        if line!='' and '+' in line:
            if 'min' in line.lower():
                obj_dir=[1] #[-1]
            a1.append(re.split(r'\++',line.lower().replace('maximize','').replace('minimize','')))
    # b1=[]
    # b1_dir=[]
    # max_col=0
    # for line in text4:
    #     if line!='':
    #         if '≥' in line:
    #             b1_dir.append(-1)
    #         else:
    #             b1_dir.append(1)

    #         splitted_line=re.split(r'\+|≥|≤',line.strip())
    #         b1.append(splitted_line)
    #         if len(splitted_line)>max_col:
    #             max_col=len(splitted_line)

    b1_rhs=[]
    b1_lhs=[]
    b1_dir=[]

    # for line in text4:
    #     # if '≥' not in line and '≤' not in line and '<' not in line and '>' not in line:
    #     #     text4.remove(line)
    #     if "x > 0"==line.stripe() or "x ≥ 0"==line.stripe() or "y > 0"==line.stripe() or "y ≥ 0"==line.stripe():
            
    #         text4.remove(line)
    #     if "x, y > 0"==line.stripe() or "x, y ≥ 0"==line.stripe() or "y, x > 0"==line.stripe() or "y, x ≥ 0"==line.stripe():
    #         print('yesss')
    #         text4.remove(line)

    for line in text4:
        line=line.replace("-","+ -")
        if '≥' in line or '≤' in line or '<' in line or '>' in line:
            if '≥' in line:
                b1_dir.append(-1)
                idx=line.find('≥')
            elif '≤' in line:
                b1_dir.append(1)
                idx=line.find('≤')
            elif '<' in line:
                b1_dir.append(1)
                idx=line.find('<')
            elif '>' in line:
                b1_dir.append(1)
                idx=line.find('>')            
            splitted_line_lhs=re.split(r'\+',line[:idx].strip())
            splitted_line_rhs=re.split(r'\+',line[idx+1:].strip())
            b1_rhs.append(splitted_line_rhs)
            b1_lhs.append(splitted_line_lhs)




    canon_form_obj=np.ones(max_col-1)

    # obj_dir=[-1 if 'min' in a1[0][0].lower() else 1]
    # print(a1)
    obj_idx=None
    try:
        for ii in range(len(a1)):
            if a1[ii][0].strip()!='':
                line=a1[ii]
    except:
        line=['x','y']


    for jj in range(len(line)):
        term=line[jj]
        if 'x' in term:
            if term.strip()=='x':
                canon_form_obj[0]=1*obj_dir[0]
            else:
                canon_form_obj[0]=float(term.strip().replace("x"," ").replace(",","").replace("$",""))*obj_dir[0]
        elif 'y' in term:
            if term.strip()=='y':
                canon_form_obj[1]=1*obj_dir[0]
            else:
                canon_form_obj[1]=float(term.strip().replace("y"," ").replace(",","").replace("$",""))*obj_dir[0]
        elif 'z' in term:
            if term.strip()=='y':
                canon_form_obj[2]=1*obj_dir[0]
            else:
                canon_form_obj[2]=float(term.strip().replace("z"," ").replace(",","").replace("$",""))*obj_dir[0]

    canon_form_const=np.zeros((len(text4),max_col))
    for ii in range(len(b1_lhs)):
        line=b1_lhs[ii]
        
        dir=b1_dir[ii]
        for jj in range(len(line)):
            term=line[jj]
            term_sign=1
            if '-' in term.strip():
                term_sign=-1
                term=term.replace("-"," ")
            if 'x' in term and 'y' in term:
                pass
            elif 'y)' in term:
                canon_form_const[ii,1]+=float(line[jj-1].strip().replace("x"," ").replace("("," ").replace(",","").replace("$",""))*dir*term_sign
            elif 'x' in term:
                if term.strip()=='x':
                    canon_form_const[ii,0]+=1*dir
                else:
                    canon_form_const[ii,0]+=float(term.strip().replace("x"," ").replace(")"," ").replace("("," ").replace(",","").replace("$",""))*dir*term_sign
            elif 'y' in term:
                if term.strip()=='y':
                    canon_form_const[ii,1]+=1*dir
                else:
                    canon_form_const[ii,1]+=float(term.strip().replace("y"," ").replace(")"," ").replace("("," ").replace(",","").replace("$",""))*dir*term_sign
            elif 'z' in term:
                if term.strip()=='z':
                    canon_form_const[ii,2]+=1*dir
                else:
                    canon_form_const[ii,2]+=float(term.strip().replace("z"," ").replace(")"," ").replace("("," ").replace(",","").replace("$",""))*dir*term_sign
            else:
                canon_form_const[ii,max_col-1]+=float(re.sub("[^0-9]", "", term))*dir

    for ii in range(len(b1_rhs)):
        line=b1_rhs[ii]
        rhs_sign=-1
        dir=b1_dir[ii]
        for jj in range(len(line)):
            term=line[jj]
            term_sign=1
            if '-' in term.strip():
                term_sign=-1
                term=term.replace("-"," ")
            if 'y)' in term:
                canon_form_const[ii,1]+=float(line[jj-1].strip().replace("x"," ").replace("("," ").replace(",","").replace("$",""))*dir*rhs_sign*term_sign
            elif 'x' in term:
                if term.strip()=='x':
                    canon_form_const[ii,0]+=1*dir*rhs_sign
                else:
                    canon_form_const[ii,0]+=float(term.strip().replace("x"," ").replace("("," ").replace(",","").replace("$",""))*dir*rhs_sign*term_sign
            elif 'y' in term:
                if term.strip()=='y':
                    canon_form_const[ii,1]+=1*dir*rhs_sign
                else:
                    canon_form_const[ii,1]+=float(term.strip().replace("y"," ").replace("("," ").replace(",","").replace("$",""))*dir*rhs_sign*term_sign
            elif 'z' in term:
                if term.strip()=='z':
                    canon_form_const[ii,2]+=1*dir*rhs_sign
                else:
                    canon_form_const[ii,2]+=float(term.strip().replace("z"," ").replace("("," ").replace(",","").replace("$",""))*dir*rhs_sign*term_sign
            elif re.sub("[^0-9]", "", term)!='':
                canon_form_const[ii,max_col-1]+=float(re.sub("[^0-9]", "", term))*dir
            
    # print(canon_form_obj, canon_form_const)
    
    return canon_form_obj, canon_form_const

def energy_score(logits):
    return torch.logsumexp(logits, dim=-1)

import torch
import torch.nn.functional as F


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce #.mean()
        return loss
    

def eval_equation(equation):
    try:
        answer = eval(equation)
    except:
        answer = np.nan

    return answer


def remove_letters(input_string):
    return ''.join(char for char in input_string if char.isdigit() or char in ('(', ')','-','+','/','.','*'))