"""Partner outcomes from existing tournament CSV records. No simulation imports."""
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def summarize_partners(archive: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts=[]
    total=0

    def process(records):
        nonlocal total
        if records.empty:
            return
        total+=len(records)
        assert not records.duplicated(['cell_id','seed','seat']).any()
        married=records[records.married==1].merge(
            records[['cell_id','seed','seat','group','partner','happiness']],
            left_on=['cell_id','seed','partner'],right_on=['cell_id','seed','seat'],
            suffixes=('','_spouse'),validate='one_to_one',
        )
        assert len(married)==int(records.married.sum())
        assert (married.seat==married.partner_spouse).all()
        married['cohort']=np.where(married.group==married.group_spouse,'same-group','other-group')
        married['own_negative']=married.happiness<0
        married['partner_negative']=married.happiness_spouse<0
        married['own_positive_partner_negative']=(married.happiness>0)&(married.happiness_spouse<0)
        married['own_negative_partner_positive']=(married.happiness<0)&(married.happiness_spouse>0)
        married['both_positive']=(married.happiness>0)&(married.happiness_spouse>0)
        parts.append(married.groupby(['group','cohort']).agg(
            married=('seat','size'),own_happiness=('happiness','sum'),partner_happiness=('happiness_spouse','sum'),
            own_negative=('own_negative','sum'),partner_negative=('partner_negative','sum'),
            own_positive_partner_negative=('own_positive_partner_negative','sum'),
            own_negative_partner_positive=('own_negative_partner_positive','sum'),both_positive=('both_positive','sum'),
        ))

    # A run can straddle a CSV chunk. Carry its last block into the next chunk
    # so each partner lookup always has the complete population available.
    carry=None
    with zipfile.ZipFile(archive) as z:
        for chunk in pd.read_csv(
            z.open('tuned_p10_tournament_results/7_raw_runs.csv'),
            usecols=['cell_id','seed','seat','group','partner','married','happiness','positive'],
            dtype={'group':str},chunksize=300000,
        ):
            if carry is not None:
                chunk=pd.concat([carry,chunk],ignore_index=True)
            last=(chunk.cell_id.iloc[-1],chunk.seed.iloc[-1])
            tail=(chunk.cell_id==last[0])&(chunk.seed==last[1])
            carry=chunk[tail]
            process(chunk[~tail])
        process(carry)
    assert total==5510904
    by_cohort=pd.concat(parts).groupby(level=[0,1]).sum()
    overall=by_cohort.groupby(level='group').sum()
    for frame in [by_cohort,overall]:
        for col in ['own_happiness','partner_happiness','own_negative','partner_negative',
                    'own_positive_partner_negative','own_negative_partner_positive','both_positive']:
            frame[col+'_mean']=frame[col]/frame.married
        frame['own_minus_partner_mean']=frame.own_happiness_mean-frame.partner_happiness_mean
    # Self-group pairs are counted from both seats: score totals must agree.
    same=by_cohort.xs('same-group',level='cohort')
    assert np.allclose(same.own_happiness,same.partner_happiness)
    assert (same.own_negative==same.partner_negative).all()
    return overall,by_cohort
