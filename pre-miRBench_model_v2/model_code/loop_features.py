"""Deterministic loop-aligned features. No labels, fitting, or file access."""
import numpy as np
META = ['presence','opening_div200','closing_div200','midpoint_div200','interior_length_div200','absolute_midpoint_offset_div200','post_open_depth_div200','ancestor_count_including_self_div200']
STEM = ['presence'] + ['open_nt_'+b for b in 'ACGU'] + ['close_nt_'+b for b in 'ACGU'] + ['pair_'+p for p in ('AU','UA','GC','CG','GU','UG')] + ['opening_div200','closing_div200','span_div200','signed_midpoint_offset_div200','opening_gap_div200','closing_gap_div200','stack_length_div200']
APPENDED_NAMES = []
for rank in range(3):
    prefix=f'loop{rank+1}'
    APPENDED_NAMES += [f'{prefix}.meta.{s}' for s in META]
    APPENDED_NAMES += [f'{prefix}.relative.{offset:+d}.{s}' for offset in range(-20,21) for s in ['nt_A','nt_C','nt_G','nt_U','db_dot','db_open','db_close']]
    APPENDED_NAMES += [f'{prefix}.ancestor{a:02d}.{s}' for a in range(32) for s in STEM]
assert len(APPENDED_NAMES)==2997 and len(STEM)==22

def parse(sequence, structure):
    if not isinstance(sequence,str) or not isinstance(structure,str):
        raise ValueError('Sequence and structure must be strings')
    sequence=sequence.upper().replace('T','U')
    if len(sequence)!=200 or len(structure)!=200 or set(sequence)-set('ACGUN'):
        raise ValueError('Require 200 nt, alphabet ACGUN after uppercase/T-to-U')
    partner=np.full(200,-1,dtype=np.int64); depth=np.zeros(200,dtype=np.int64)
    parent={}; children={}; opens=[]
    for i,c in enumerate(structure):
        if c=='(':
            parent[i]=opens[-1] if opens else None
            children[i]=[]
            if opens: children[opens[-1]].append(i)
            opens.append(i)
        elif c==')':
            if not opens: raise ValueError('Unbalanced closing parenthesis')
            j=opens.pop(); partner[i]=j; partner[j]=i
        elif c!='.': raise ValueError('Unsupported dot-bracket symbol')
        depth[i]=len(opens)
    if opens: raise ValueError('Unbalanced opening parenthesis')
    stacks={}
    for i in children:
        j=int(partner[i])
        if i>0 and j<199 and partner[i-1]==j+1: continue
        n=1
        while i+n<j-n and partner[i+n]==j-n: n+=1
        for k in range(n): stacks[i+k]=n
    loops=sorted((i for i in children if not children[i]),key=lambda i:(abs((i+int(partner[i]))/2-99.5),i,int(partner[i])))
    return sequence,partner,depth,parent,children,stacks,loops

def loop_features(sequence,structure):
    seq,partner,depth,parent,children,stacks,loops=parse(sequence,structure)
    out=np.zeros((3,999),dtype=np.float32)
    for rank,i in enumerate(loops[:3]):
        j=int(partner[i]); chain=[]; a=i
        while a is not None: chain.append(a); a=parent[a]
        out[rank,:8]=[1,i/200,j/200,(i+j)/400,(j-i-1)/200,abs((i+j)/2-99.5)/200,depth[i]/200,len(chain)/200]
        relative=out[rank,8:295].reshape(41,7); anchor=(i+j)//2
        for t,offset in enumerate(range(-20,21)):
            p=anchor+offset
            if 0<=p<200:
                for b,base in enumerate('ACGU'): relative[t,b]=seq[p]==base
                for b,c in enumerate('.()'): relative[t,4+b]=structure[p]==c
        stem=out[rank,295:].reshape(32,22)
        for k,a in enumerate(chain[:32]):
            b=int(partner[a]); gap_l=gap_r=0
            if k:
                inner=chain[k-1]; gap_l=inner-a-1; gap_r=b-int(partner[inner])-1
            stem[k]=[1]+[int(seq[a]==c) for c in 'ACGU']+[int(seq[b]==c) for c in 'ACGU']+[int(seq[a]+seq[b]==p) for p in ('AU','UA','GC','CG','GU','UG')]+[a/200,b/200,(b-a)/200,((a+b)/2-99.5)/200,gap_l/200,gap_r/200,stacks[a]/200]
    return out.ravel()

def extract_appended(samples):
    return np.asarray([loop_features(s,t) for s,t in zip(samples.sequence_rna,samples.structure)],dtype=np.float32).reshape(len(samples),2997)
