"""Conservative numeric equation checks. Uses a bounded AST interpreter, never eval."""
import ast,re,math
from fractions import Fraction
import numpy as np
from common import parse_answer

NAMES=['log_checked','log_bad','log_good','bad_fraction','has_bad','max_relative_error','mean_relative_error','last_bad','final_bad','final_good','final_seen','division_fraction','log_raw_equalities','coverage','exact_fraction','bad_fraction_vs_group','bad_count_vs_group','answer_group_bad_fraction']

def normalize(text):
    s=re.sub(r'\*\*([^*\n]+)\*\*',r'\1',text)
    s=s.replace('×','*').replace('÷','/').replace('−','-').replace('–','-')
    s=re.sub(r'\\(?:times|cdot)\b','*',s); s=re.sub(r'\\div\b','/',s)
    s=re.sub(r'\\(?:left|right)\b','',s)
    s=re.sub(r'\\(?:text|mathrm|operatorname)\{[^{}]*\}',' UNIT ',s)
    s=re.sub(r'\\(?:begin|end)\{[^{}]*\}',' ',s)
    for _ in range(6):
        newer=re.sub(r'\\(?:dfrac|tfrac|frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}',lambda m:'(('+m[1]+')/('+m[2]+'))',s)
        if newer==s: break
        s=newer
    s=re.sub(r'\\boxed\{([^{}]*)\}',r'(\1)',s)
    s=s.replace('\\\\','\n').replace('\\[',' ').replace('\\]',' ').replace('\\(',' ').replace('\\)',' ')
    s=re.sub(r'\\[,;:! ]',' ',s).replace('\\$',' ').replace('$',' ').replace('&',' ')
    s=s.replace('\\%','%').replace('{','(').replace('}',')').replace('^','**')
    s=re.sub(r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?',lambda m:m[0].replace(',',''),s)
    s=re.sub(r'(\d+(?:\.\d+)?)\s*%',r'(\1/100)',s)
    return s

def number_value(expression):
    src=expression.strip()
    if not src or len(src)>180 or not re.fullmatch(r'[\d.\s()+*/\-]+',src): raise ValueError('Not a bounded numeric expression')
    tree=ast.parse(src,mode='eval')
    def walk(node,depth=0):
        if depth>20: raise ValueError('Too deep')
        if isinstance(node,ast.Expression): return walk(node.body,depth+1)
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            literal=ast.get_source_segment(src,node)
            if len(literal)>40: raise ValueError('Too large')
            value=Fraction(literal)
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value=walk(node.operand,depth+1)*(1 if isinstance(node.op,ast.UAdd) else -1)
        elif isinstance(node,ast.BinOp):
            left=walk(node.left,depth+1); right=walk(node.right,depth+1)
            if isinstance(node.op,ast.Add):value=left+right
            elif isinstance(node.op,ast.Sub):value=left-right
            elif isinstance(node.op,ast.Mult):value=left*right
            elif isinstance(node.op,ast.Div):value=left/right
            elif isinstance(node.op,ast.Pow) and right.denominator==1 and abs(right)<=8:value=left**int(right)
            else:raise ValueError('Unsupported operation')
        else:raise ValueError('Unsupported syntax')
        if value.numerator.bit_length()>256 or value.denominator.bit_length()>256:raise ValueError('Too large')
        return value
    return walk(tree)

def _side(segment,tail):
    segment=segment.strip()
    if tail and re.match(r'^[-*]\s+\d',segment):return None
    pattern=r'([\d.\s()+*/\-]+)$' if tail else r'^([\d.\s()+*/\-]+)'
    match=re.search(pattern,segment)
    if not match:return None
    expression=match[1].strip()
    if expression.startswith(('+','*','/')):return None
    if tail and match.start()>0 and segment[match.start()-1] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_':
        # Avoid treating a numeric suffix of an identifier such as x2 as a scalar.
        if not segment[match.start()].isspace():return None
    if tail:
        prefix=segment[:match.start()].strip()
        if re.search(r'[\d+*/=]',prefix) or re.search(r'\b[A-Za-z]$',prefix):return None
    if not tail:
        remainder=segment[match.end():].lstrip()
        if remainder.startswith(('**','/','*','+','-','(')):return None
        if remainder and match.end()<len(segment) and segment[match.end()].isalpha():return None
    try:return expression,number_value(expression)
    except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,TypeError):return None

def equations(text):
    records=[]
    for line in normalize(text).splitlines():
        if len(line)>5000:continue
        pieces=line.split('=')
        for left,right in zip(pieces,pieces[1:]):
            a=_side(left,True); b=_side(right,False)
            if a is None or b is None:continue
            expr_a,va=a; expr_b,vb=b
            if not any(op in expr_a+expr_b for op in ['+','*','/']) and '-' not in expr_a.lstrip('-')+expr_b.lstrip('-'):continue
            av=float(va); bv=float(vb)
            if not math.isfinite(av) or not math.isfinite(bv):continue
            difference=abs(av-bv); scale=max(abs(av),abs(bv),1e-8); tolerance=1e-6+.01*scale
            literal=re.fullmatch(r'[-+]?\d*\.(\d{1,8})',expr_b)
            if literal:tolerance=max(tolerance,.5*10**(-len(literal[1]))+1e-9)
            records.append({'left':expr_a,'right':expr_b,'left_value':av,'right_value':bv,'bad':difference>tolerance,'relative_error':min(difference/scale,10.),'exact':va==vb,'division':'/' in expr_a+expr_b})
    return records

def candidate_stats(text):
    records=equations(text); n=len(records); bad=sum(r['bad'] for r in records); final=parse_answer(text); final=float(Fraction(final)) if final is not None else None
    final_records=[r for r in records if final is not None and abs(r['right_value']-final)<1e-6]
    errors=[r['relative_error'] for r in records]
    result=[np.log1p(n),np.log1p(bad),np.log1p(n-bad),bad/max(n,1),float(bad>0),max(errors,default=0),sum(errors)/max(n,1),float(records[-1]['bad']) if records else 0.,sum(r['bad'] for r in final_records),sum(not r['bad'] for r in final_records),float(bool(final_records)),sum(r['division'] for r in records)/max(n,1),np.log1p(text.count('=')),n/max(text.count('='),1),sum(r['exact'] for r in records)/max(n,1)]
    return result,records

def arithmetic_features(bank):
    output=[]
    for row in bank:
        rows=[candidate_stats(c['text'])[0] for c in row['candidates']]; values=np.asarray(rows,dtype=np.float32); answers=[parse_answer(c['text']) for c in row['candidates']]
        for j,x in enumerate(rows):
            peers=[k for k,a in enumerate(answers) if a is not None and a==answers[j]] or [j]
            output.append(x+[float(values[j,3]-values[:,3].mean()),float(values[j,1]-values[:,1].mean()),float(values[peers,3].mean())])
    return np.asarray(output,dtype=np.float32)
