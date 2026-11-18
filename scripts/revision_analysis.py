"""Reproduce the additional reviewer analyses without overwriting archived results.
Run from the repository root: python scripts/revision_analysis.py [stats|train|generate].
Intervals describe repeated splits of one population, not engineering uncertainty.
"""
from pathlib import Path
import sys, json, platform
import numpy as np
import pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_config, load_dataset, FEATURE_COLUMNS, ci95
OUT = ROOT / 'revision_results'
OUT.mkdir(exist_ok=True)
REF = ROOT / 'reference_results' / 'tables'

def static_stats():
    baseline = pd.read_csv(REF / 'baseline_detail.csv')
    ablation = pd.read_csv(REF / 'ablation_detail.csv')
    rf = baseline[baseline.model == 'Full (28) + Random forest']
    modal = ablation[ablation.feature_set == 'Cumulative: 12 features']
    paired = modal.merge(rf, on=['repeat','seed'], suffixes=('_modal','_rf'), validate='one_to_one')
    d = 100*(paired.test_accuracy_modal-paired.test_accuracy_rf)
    paired['difference_pp'] = d
    paired.to_csv(OUT/'paired_modal_rf.csv',index=False)
    n=len(d); sd=float(d.std(ddof=1)); mean=float(d.mean())
    # Corrected resampled t approximation accounts for train/test overlap.
    fraction=load_config()['validation']['test_fraction']
    se=sd*np.sqrt(1/n+fraction/(1-fraction))
    half=stats.t.ppf(.975,n-1)*se
    result={'n':n,'mean_difference_pp':mean,'sd_difference_pp':sd,
            'corrected_se_pp':se,'corrected_t':mean/se,
            'corrected_p_two_sided':float(2*stats.t.sf(abs(mean/se),n-1)),
            'corrected_95_low_pp':mean-half,'corrected_95_high_pp':mean+half,
            'interpretation':'Approximate repeated-split comparison; not evidence of equivalence or external validity.'}
    (OUT/'paired_test.json').write_text(json.dumps(result,indent=2))
    data=load_dataset()
    data.groupby(['damage_story','nominal_stiffness_loss']).freq_1_hz.agg(['mean','std','min','max']).to_csv(OUT/'frequency_summary.csv')
    print(result,flush=True)

def train():
    import sklearn, scipy
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, f1_score
    from src.run_validation import make_mlp, read_settings, evaluate
    cfg=load_config(); data=load_dataset(); y=data.label_id.to_numpy(dtype=int)
    recorded=pd.read_csv(REF/'split_metrics.csv').set_index('seed')
    archived_class_f1=pd.read_csv(REF/'per_class_f1_detail.csv')
    class_f1_match=True
    pooled=np.zeros((13,13),dtype=int); rows=[]; predrows=[]; checks=[]
    for repeat, seed in enumerate(cfg['validation']['seeds'],1):
        tr,te=train_test_split(np.arange(len(y)),test_size=cfg['validation']['test_fraction'],random_state=seed,stratify=y)
        r=evaluate(make_mlp(seed,read_settings(cfg)),data[FEATURE_COLUMNS].iloc[tr],data[FEATURE_COLUMNS].iloc[te],y[tr],y[te])
        yp=r['test_pred']; yt=y[te]; pooled+=confusion_matrix(yt,yp,labels=range(13))
        old_f1=archived_class_f1[archived_class_f1.seed==seed].sort_values('label_id').f1.to_numpy()
        class_f1_match=class_f1_match and bool(np.allclose(f1_score(yt,yp,labels=range(13),average=None,zero_division=0),old_f1,rtol=0,atol=1e-12))
        checks.append({'seed':seed,'test_accuracy':r['test_accuracy'],'archived_test_accuracy':recorded.loc[seed,'test_accuracy'],
                       'macro_f1':r['test_macro_f1'],'archived_macro_f1':recorded.loc[seed,'test_macro_f1'],
                       'stopping_epoch':r['n_iter'],'archived_stopping_epoch':int(recorded.loc[seed,'actual_stopping_epoch'])})
        st=lambda a: np.where(a>0,1+(a-1)//3,0)
        lv=lambda a: np.where(a>0,1+(a-1)%3,0)
        row={'repeat':repeat,'seed':seed}
        for level in [1,2,3]:
            mask=lv(yt)==level
            row[f'location_at_{level*10}']=float(np.mean(st(yp)[mask]==st(yt)[mask]))
        for story in [1,2,3,4]:
            mask=st(yt)==story
            row[f'severity_at_story_{story}']=float(np.mean(lv(yp)[mask]==lv(yt)[mask]))
        rows.append(row)
        predrows.extend({'repeat':repeat,'seed':seed,'dataset_row_zero_based':int(i),'true_class':int(a),'predicted_class':int(b)} for i,a,b in zip(te,yt,yp))
        print('Split',repeat,'accuracy',r['test_accuracy'],'archived',recorded.loc[seed,'test_accuracy'],flush=True)
    pd.DataFrame(checks).to_csv(OUT/'rerun_checks.csv',index=False)
    pd.DataFrame(predrows).to_csv(OUT/'split_predictions.csv',index=False)
    frame=pd.DataFrame(rows); frame.to_csv(OUT/'factorized_detail.csv',index=False)
    summary=[]
    for col in frame.columns[2:]:
        m,s,lo,hi=ci95(100*frame[col]);summary.append({'metric':col,'mean_pct':m,'sd_pct':s,'ci95_low_pct':lo,'ci95_high_pct':hi})
    pd.DataFrame(summary).to_csv(OUT/'factorized_summary.csv',index=False)
    archived=pd.read_csv(REF/'pooled_confusion_matrix.csv',index_col=0).to_numpy(dtype=int)
    pd.DataFrame(pooled).to_csv(OUT/'rerun_pooled_confusion.csv',index=False)
    valid={'pooled_exact_match':bool(np.array_equal(pooled,archived)),
           'all_class_f1_match':class_f1_match,
           'all_split_metrics_match':all(np.isclose(r['test_accuracy'],r['archived_test_accuracy'],rtol=0,atol=1e-12) and np.isclose(r['macro_f1'],r['archived_macro_f1'],rtol=0,atol=1e-12) and r['stopping_epoch']==r['archived_stopping_epoch'] for r in checks),
           'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'platform':platform.platform()}
    (OUT/'rerun_validation.json').write_text(json.dumps(valid,indent=2));print(valid,flush=True)

def generate():
    from src.generate_dataset import generate as generate_data
    records=[]; generated=generate_data(records); original=load_dataset()
    numeric=original.select_dtypes('number').columns
    equal=bool(np.allclose(generated[numeric],original[numeric],rtol=1e-9,atol=1e-12)) and generated.label_name.equals(original.label_name)
    if not equal: raise RuntimeError('Regenerated data do not match the archived dataset')
    frame=pd.DataFrame(records)
    # These records are in generation order; the original dataset is shuffled.
    frame.insert(0,'generation_row_zero_based',np.arange(len(frame)))
    frame.to_csv(OUT/'actual_damage_generation_order.csv',index=False)
    frame.groupby('nominal_stiffness_loss').actual_stiffness_loss.agg(['count','min','max','mean','std']).to_csv(OUT/'actual_damage_ranges.csv')
    frame.groupby(['damage_story','nominal_stiffness_loss']).actual_stiffness_loss.agg(['count','min','max']).to_csv(OUT/'actual_damage_ranges_by_story.csv')
    result={'regeneration_matches_archive':equal,'max_abs_error':float(np.max(np.abs(generated[numeric].to_numpy()-original[numeric].to_numpy()))),'records_order':'generation order before original shuffle; no additional random draws'}
    (OUT/'regeneration_audit.json').write_text(json.dumps(result,indent=2));print(result,flush=True)

if __name__=='__main__':
    {'stats':static_stats,'train':train,'generate':generate}[sys.argv[1] if len(sys.argv)>1 else 'stats']()
