"""ML mixed-effects LRT for Group, State and Group:State from ep.csv.
python test_ep_lrt.py results/beta/ep.csv --output results/beta/ep_lrt_results.csv
Requires pandas scipy statsmodels numpy.
"""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests


def run(csv_file):
    df = pd.read_csv(csv_file)
    required={'group','patient_id','state','band','model','ep_normal','ep_conjugate'}
    missing=required-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    df['subject']=df['group'].astype(str)+'_'+df['patient_id'].astype(str)
    df['state']=pd.Categorical(df['state'],categories=['baseline','mild','moderate','recovery'])
    out=[]
    for (band, model), base in df.groupby(['band','model']):
        for ep in ['ep_normal','ep_conjugate']:
            d=base[['subject','group','state',ep]].dropna().copy()
            if d.subject.nunique()<5: continue
            terms={'group':f'{ep} ~ C(group)', 'state':f'{ep} ~ C(state)',
                   'add':f'{ep} ~ C(group)+C(state)',
                   'full':f'{ep} ~ C(group)*C(state)'}
            fits={}
            for name, formula in terms.items():
                for method in ['lbfgs','powell','bfgs']:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            fit=mixedlm(formula,d,groups=d.subject).fit(reml=False,method=method,maxiter=2000,disp=False)
                        if fit.converged and np.isfinite(fit.llf):
                            fits[name]=fit
                            break
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                if name not in fits: raise RuntimeError(f'MixedLM did not converge: {band}/{model}/{ep}/{name}')
            for effect, reduced, full in [('group','state','add'),('state','group','add'),('interaction','add','full')]:
                a,b=fits[reduced],fits[full]
                LR=2*(b.llf-a.llf)
                delta_df=len(b.fe_params)-len(a.fe_params)
                out.append({'band':band,'model':model,'ep_type':ep,'effect':effect,
                            'n_subjects':d.subject.nunique(),'n_rows':len(d),
                            'lr_statistic':LR,'df':delta_df,'p_value':chi2.sf(max(LR,0),delta_df),
                            'comparison':f'{reduced} vs {full}'})
    result=pd.DataFrame(out)
    if not result.empty:
        result['p_fdr_bh']=multipletests(result.p_value,method='fdr_bh')[1]
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('--output',default='ep_lrt_results.csv')
    args=parser.parse_args()
    r=run(args.csv)
    r.to_csv(args.output,index=False)
    print(r.to_string(index=False))
