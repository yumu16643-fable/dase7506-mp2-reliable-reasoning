"""Reference-free candidate features; shared across all candidate positions."""
import re
from collections import Counter
from fractions import Fraction
import numpy as np
from common import parse_answer

NUM = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
PATTERNS = [r'\btherefore\b', r'\bthus\b', r'\bso\b', r'\bhowever\b', r'\bbut\b', r'\bwait\b', r'\bwrong\b|\bincorrect\b', r'\bcheck\b|\bverify\b', r'\bassum', r'\bapproximately\b|\bapprox', r'\btotal\b', r'\bremaining\b|\bleft\b', r'\bhalf\b', r'\beach\b|\bper\b', r'\banswer\b', r'\bstep\b', r'\bnot\b', r'\bif\b', r'\bmaybe\b|\bmight\b|\bperhaps\b']
NAMES = ['log_chars','log_words','log_lines','log_numbers','equals','boxed','hash_answer','bold','latex','digits_fraction','question_words','valid','negative','zero','integer','log_answer_magnitude','answer_in_question']
NAMES += ['word_' + str(i) for i in range(len(PATTERNS))]
NAMES += ['group_count','group_fraction','valid_fraction','distinct_answers','max_agreement','majority_member','unique_answer','answer_z','relative_length','relative_numbers']

def text_features(bank):
    result=[]
    for row in bank:
        texts=[c['text'] for c in row['candidates']]
        answers=[parse_answer(t) for t in texts]
        counts=Counter(a for a in answers if a is not None)
        nums=[float(Fraction(a)) if a is not None else 0. for a in answers]
        signed=np.sign(nums)*np.log1p(np.abs(nums))
        qnums=set(NUM.findall(row['question'].replace(',','')))
        lengths=[len(t) for t in texts]; numcounts=[len(NUM.findall(t)) for t in texts]
        valid=sum(a is not None for a in answers); maximum=max(counts.values(),default=0)
        for j,(t,a) in enumerate(zip(texts,answers)):
            val=nums[j]
            base=[np.log1p(len(t)),np.log1p(len(t.split())),np.log1p(t.count('\n')),np.log1p(numcounts[j]),min(t.count('='),50)/10,min(t.count('\\boxed'),10),min(t.count('####'),10),min(t.count('**'),100)/10,min(t.count('\\'),200)/20,sum(c.isdigit() for c in t)/max(len(t),1),np.log1p(len(row['question'].split())),float(a is not None),float(val<0),float(a is not None and val==0),float(a is not None and Fraction(a).denominator==1),np.log1p(abs(val)),float(a is not None and a in qnums)]
            base += [min(len(re.findall(p,t.lower())),20)/5 for p in PATTERNS]
            count=counts.get(a,0) if a is not None else 0
            base += [count,count/4,valid/4,len(counts),maximum,float(count==maximum and count>0),float(count==1),float((signed[j]-np.mean(signed))/(np.std(signed)+1)),len(t)/(np.mean(lengths)+1),numcounts[j]/(np.mean(numcounts)+1)]
            result.append(base)
    return np.asarray(result,dtype=np.float32)

def make_features(bank,arrays,mode):
    e=np.asarray(arrays['embeddings'],dtype=np.float32); n=len(bank)
    if mode=='embedding': return e.reshape(4*n,-1)
    t=text_features(bank)
    if mode=='text': return t
    if mode=='style': return t[:,:11]
    if mode=='agreement': return t[:,[11]+list(range(len(NAMES)-10,len(NAMES)-2))]
    if mode=='combined': return np.concatenate([e.reshape(4*n,-1),t],axis=1)
    if mode=='context': return np.concatenate([e.reshape(4*n,-1),(e-e.mean(axis=1,keepdims=True)).reshape(4*n,-1),t],axis=1)
    raise ValueError(mode)

def choose(scores,bank):
    scores=np.asarray(scores,dtype=np.float64).reshape(len(bank),4).copy()
    valid=np.array([[parse_answer(c['text']) is not None for c in r['candidates']] for r in bank])
    scores[~valid & valid.any(axis=1,keepdims=True)]=-1e12
    return np.argmax(scores,axis=1).astype(int).tolist()

def majority(bank):
    values=[]
    for row in bank:
        answers=[parse_answer(c['text']) for c in row['candidates']]
        counts=Counter(a for a in answers if a is not None)
        values.extend([counts.get(a,0) if a is not None else -1 for a in answers])
    return np.array(values).reshape(-1,4)
