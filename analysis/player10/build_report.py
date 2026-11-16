"""Reproduce the Player 10 report from the tournament ZIP (no simulations).

Requires pandas, numpy, matplotlib, reportlab. Run from repository root.
The raw CSV is streamed; only per-run aggregates and Player 10 seats are retained.
"""
from pathlib import Path
import argparse
import hashlib
import sys
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--deps', help='Optional additional Python package directory')
parser.add_argument('--cached', action='store_true', help='Reuse /private/tmp/p10_*.pkl')
args = parser.parse_args()
if args.deps:
    sys.path.append(args.deps)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/pdf'
FIG = OUT / 'player10_figures'
TAB = OUT / 'player10_tables'
for p in [OUT, FIG, TAB]: p.mkdir(parents=True, exist_ok=True)
archive = ROOT / 'tuned_p10_tournament_results.zip'
z = zipfile.ZipFile(archive)
if args.cached:
    r = pd.read_pickle('/private/tmp/p10_runs.pkl')
    f = pd.read_pickle('/private/tmp/p10_seats.pkl')
else:
    parts, focal, equal_raw_parts = [], [], []
    keys = ['cell_id','seed','slice','roster','bonus','rounds','dims','monotonic','distribution','people','group']
    cols = keys + ['seat','married','happiness','positive','partner','married_on','faults']
    for c in pd.read_csv(z.open('tuned_p10_tournament_results/7_raw_runs.csv'), usecols=cols, chunksize=250000, dtype={'group':str}):
        c['negative'] = (c.happiness < 0).astype(int)
        c['zero_marriage'] = ((c.happiness == 0) & (c.married == 1)).astype(int)
        parts.append(c.groupby(keys, observed=True).agg(seats=('seat','size'),positive=('positive','sum'),married=('married','sum'),happiness=('happiness','sum'),negative=('negative','sum'),zero_marriage=('zero_marriage','sum'),faults=('faults','max')).reset_index())
        focal.append(c[c.group == '10'])
        equal_raw_parts.append(c[c.roster == 'equal'])
    r = pd.concat(parts).groupby(keys, as_index=False).agg({'seats':'sum','positive':'sum','married':'sum','happiness':'sum','negative':'sum','zero_marriage':'sum','faults':'max'})
    f = pd.concat(focal, ignore_index=True)
    r.to_pickle('/private/tmp/p10_runs.pkl'); f.to_pickle('/private/tmp/p10_seats.pkl')
    pd.concat(equal_raw_parts,ignore_index=True).to_pickle('/private/tmp/p10_equal_seats.pkl')

standing = pd.read_csv(z.open('tuned_p10_tournament_results/1_overall_standings.csv'))
assert len(f) == 405540
assert len(r[['cell_id','seed']].drop_duplicates()) == 80520
assert r.seats.sum() == 5510904 and r.faults.max() == 0
for row in standing.itertuples():
    a = r[r.group == str(row.group)]
    assert a.seats.sum() == row.seats
    for col, val in [('positive',row.positive_rate),('married',row.married_rate),('happiness',row.mean_happiness)]:
        assert abs(a[col].sum()/a.seats.sum()-val) < 0.00000051

def rates(a):
    a = a.copy()
    for src, dest in [('positive','positive_rate'),('married','married_rate'),('happiness','mean_happiness'),('negative','negative_rate')]:
        a[dest] = a[src] / a.seats
    return a

metrics = ['seats','positive','married','happiness','negative','zero_marriage']
roster = rates(r[r.group == '10'].groupby('roster')[metrics].sum())
equal = rates(r[(r.roster == 'equal') & (r.group != 'r')].groupby('group')[metrics].sum()).sort_values('positive_rate', ascending=False)
equal['rank'] = np.arange(1, len(equal)+1)
f['mode'] = np.select([f.monotonic == 0, f.monotonic == f.dims], ['none','all'], default='half')
dm = f.groupby(['dims','mode']).agg(seats=('seat','size'),positive_rate=('positive','mean'),married_rate=('married','mean'),negative_rate=('negative','mean'))
bd = f.groupby(['dims','bonus']).agg(seats=('seat','size'),positive_rate=('positive','mean'),negative_rate=('negative','mean'))
dist = f.groupby('distribution').agg(seats=('seat','size'),positive_rate=('positive','mean'),married_rate=('married','mean'),negative_rate=('negative','mean'),mean_happiness=('happiness','mean'))
x = r[r.roster.isin(['half-half','half-half_sample'])]
h = x[x.group == '10'].merge(x[(x.group != '10') & (x.group != 'r')], on=['cell_id','seed'], suffixes=('_10','_opp'))
head = []
for g, a in h.groupby('group_opp'):
    n = a.seats_10.sum()
    assert (a.seats_10 == a.seats_opp).all()
    head.append(dict(opponent=int(g),runs=len(a),seats=n,positive_10=a.positive_10.sum()/n,positive_opp=a.positive_opp.sum()/n,gap_pp=100*(a.positive_10.sum()-a.positive_opp.sum())/n,happiness_gap=(a.happiness_10.sum()-a.happiness_opp.sum())/n))
head = pd.DataFrame(head).sort_values('opponent')
for name, a in [('overall',standing),('roster',roster),('equal_market',equal),('dimension_monotonic',dm),('dimension_bonus',bd),('distribution',dist),('head_to_head',head)]:
    a.to_csv(TAB / f'{name}.csv',index=name not in ['overall','head_to_head'])

# Retain equal-group diagnostics for explicitly labeled claims 2 and 9.
GROUPS=[str(i) for i in range(1,11)]
OTHERS=GROUPS[:-1]
er=r[(r.roster=='equal')&(r.group!='r')].copy()
er['mode']=np.select([er.monotonic==0,er.monotonic==er.dims],['none','all'],default='half')
assert er.groupby('group').seats.sum().nunique()==1
assert er.groupby('group').seats.sum().iloc[0]==14448
axes_data={}
for axis in ['mode','dims','bonus','distribution','rounds']:
    a=rates(er.groupby([axis,'group'])[metrics].sum())
    a['positive_rank']=a.groupby(level=0).positive_rate.rank(ascending=False,method='min')
    a['happiness_rank']=a.groupby(level=0).mean_happiness.rank(ascending=False,method='min')
    # Require identical seat denominators within each condition for every group.
    assert (a.seats.unstack().nunique(axis=1)==1).all()
    axes_data[axis]=a
    a.to_csv(TAB/f'individual_equal_{axis}.csv')

def gaps(axis,metric):
    v=axes_data[axis][metric].unstack().reindex(columns=GROUPS)
    return v[OTHERS].rsub(v['10'],axis=0)

for axis in axes_data:
    for metric in ['positive_rate','negative_rate','married_rate','mean_happiness']:
        gaps(axis,metric).to_csv(TAB/f'individual_gap_{axis}_{metric}.csv')

def pair_summary(a):
    rows=[]
    for g,t in a.groupby('group_opp'):
        n=t.seats_10.sum()
        assert (t.seats_10==t.seats_opp).all()
        row={'opponent':int(g),'runs':len(t),'seats_per_side':int(n)}
        for metric in ['positive','married','negative','happiness']:
            for side in ['10','opp']: row[f'{metric}_{side}']=t[f'{metric}_{side}'].sum()/n
            row[f'{metric}_gap']=row[f'{metric}_10']-row[f'{metric}_opp']
        rows.append(row)
    return pd.DataFrame(rows).set_index('opponent').sort_index()

pair=pair_summary(h)
stress=pair_summary(h[(h.dims_10==10)&(h.monotonic_10==10)])
pair.to_csv(TAB/'individual_head_to_head.csv');stress.to_csv(TAB/'individual_head_to_head_10_all.csv')
# Individual condition tables for the larger two-strategy sample as well.
for axis in ['dims_10','bonus_10','distribution_10','rounds_10']:
    rows=[]
    for k,a in h.groupby(axis):
        q=pair_summary(a).reset_index();q.insert(0,axis,k);rows.append(q)
    pd.concat(rows).to_csv(TAB/f'individual_head_to_head_by_{axis[:-3]}.csv',index=False)

cache=Path('/private/tmp/p10_timing.pkl')
if args.cached and cache.exists(): timing=pd.read_pickle(cache)
else:
    parts=[]
    for a in pd.read_csv(z.open('tuned_p10_tournament_results/7_raw_runs.csv'),usecols=['cell_id','seed','group','married','married_on','rounds'],dtype={'group':str},chunksize=300000):
        a=a[a.group!='r'].copy()
        for cut in [5,10,25,50,100]: a[f'early{cut}']=((a.married==1)&(a.married_on/a['rounds']<=cut/100)).astype(int)
        parts.append(a.groupby(['cell_id','seed','group'])[['married','early5','early10','early25','early50','early100']].sum())
    timing=pd.concat(parts).groupby(level=[0,1,2]).sum().reset_index();timing.to_pickle(cache)
check=er.merge(timing,on=['cell_id','seed','group'],suffixes=('','_timing'),validate='1:1')
assert len(check)==len(er) and (check.married==check.married_timing).all() and (check.married==check.early100).all()
tm=check.groupby('group')[['seats','married','early5','early10','early25','early50','early100']].sum().reindex(GROUPS)
for cut in [5,10,25,50,100]: tm[f'fraction{cut}']=tm[f'early{cut}']/tm.married
tm.to_csv(TAB/'individual_equal_timing.csv')

# Additional diagnostics: seed stability, downside contribution, spouse outcomes.
seed_stats=rates(er.groupby(['seed','group'])[metrics].sum())
seed_stats['rank']=seed_stats.groupby(level=0).positive_rate.rank(ascending=False,method='min')
seed_stats.to_csv(TAB/'individual_seed_stability.csv')
run_rates=er.pivot(index=['cell_id','seed'],columns='group',values='positive').div(er.pivot(index=['cell_id','seed'],columns='group',values='seats'))
run_rows=[]
for g in OTHERS:
    delta=run_rates['10']-run_rates[g]
    wins=int((delta>0).sum());ties=int((delta==0).sum());losses=int((delta<0).sum())
    assert wins+ties+losses==2394
    run_rows.append({'group':g,'wins':wins,'ties':ties,'losses':losses,'decisive_win_rate':wins/(wins+losses),'seat_gap':equal.loc['10','positive_rate']-equal.loc[g,'positive_rate']})
run_stats=pd.DataFrame(run_rows).set_index('group')
run_stats.to_csv(TAB/'individual_run_wins.csv')
raw_cache=Path('/private/tmp/p10_equal_seats.pkl')
if raw_cache.exists(): raw=pd.read_pickle(raw_cache)
else:
    raw=pd.concat([a[a.roster=='equal'] for a in pd.read_csv(z.open('tuned_p10_tournament_results/7_raw_runs.csv'),dtype={'group':str},chunksize=300000)],ignore_index=True)
    raw.to_pickle(raw_cache)
assert len(raw)==144480 and not raw.duplicated(['cell_id','seed','seat']).any()
raw['downside']=(-raw.happiness).clip(lower=0)
raw['upside']=raw.happiness.clip(lower=0)
down=raw.groupby('group').agg(seats=('seat','size'),positive=('positive','sum'),negative=('happiness',lambda a:(a<0).sum()),mean=('happiness','mean'),upside=('upside','sum'),downside=('downside','sum')).reindex(GROUPS)
down['negative_cost_per_seat']=down.downside/down.seats
down['mean_if_positive']=down.upside/down.positive
down['mean_with_negative_at_zero']=down['mean']+down.negative_cost_per_seat
assert np.allclose(down.upside-down.downside,down['mean']*down.seats)
down.to_csv(TAB/'individual_downside_decomposition.csv')
partner=raw[['cell_id','seed','seat','partner','group','happiness']].rename(columns={'seat':'partner','partner':'partner_back','group':'spouse_group','happiness':'spouse_happiness'})
couples=raw[raw.married==1].merge(partner,on=['cell_id','seed','partner'],validate='one_to_one')
assert len(couples)==int(er.married.sum()) and (couples.seat==couples.partner_back).all()
couples['own_negative']=couples.happiness<0
couples['spouse_negative']=couples.spouse_happiness<0
couples['both_positive']=(couples.happiness>0)&(couples.spouse_happiness>0)
couples['own_positive_spouse_negative']=(couples.happiness>0)&(couples.spouse_happiness<0)
spouse=couples.groupby('group').agg(married=('seat','size'),own_negative=('own_negative','sum'),spouse_negative=('spouse_negative','sum'),both_positive=('both_positive','sum'),own_positive_spouse_negative=('own_positive_spouse_negative','sum'),own_happiness=('happiness','mean'),spouse_happiness=('spouse_happiness','mean')).reindex(GROUPS)
spouse['own_negative_rate']=spouse.own_negative/spouse.married
spouse['spouse_negative_rate']=spouse.spouse_negative/spouse.married
spouse['both_positive_if_married']=spouse.both_positive/spouse.married
spouse['both_positive_all_seats']=spouse.both_positive/equal.seats
assert (spouse.married==tm.married).all()
spouse.to_csv(TAB/'individual_spouse_outcomes.csv')

# All-roster analyses: include every recorded seat of each numbered strategy.
ar=r[r.group!='r'].copy()
ar['mode']=np.select([ar.monotonic==0,ar.monotonic==ar.dims],['none','all'],default='half')
all_axes={}
for axis in ['mode','dims','distribution']:
    a=rates(ar.groupby([axis,'group'])[metrics].sum())
    a['positive_rank']=a.groupby(level=0).positive_rate.rank(ascending=False,method='min')
    all_axes[axis]=a
    a.to_csv(TAB/f'full_tournament_{axis}.csv')
    if axis in ['dims','distribution']:
        source=pd.read_csv(z.open('tuned_p10_tournament_results/'+('5_by_dims.csv' if axis=='dims' else '6_by_distribution.csv')),dtype={'group':str}).set_index([axis,'group'])
        assert (a.seats==source.seats.reindex(a.index)).all()
        for metric in ['positive_rate','married_rate','mean_happiness']:
            assert np.allclose(a[metric],source[metric].reindex(a.index),atol=.00000051,rtol=0)
all_check=ar.merge(timing,on=['cell_id','seed','group'],suffixes=('','_timing'),validate='1:1')
assert len(all_check)==len(ar) and (all_check.married==all_check.married_timing).all() and (all_check.married==all_check.early100).all()
all_tm=all_check.groupby('group')[['seats','married','early5','early10','early25','early50','early100']].sum().reindex(GROUPS)
for cut in [5,10,25,50,100]:all_tm[f'fraction{cut}']=all_tm[f'early{cut}']/all_tm.married
all_tm.to_csv(TAB/'full_tournament_timing.csv')
all_totals=rates(ar.groupby('group')[metrics].sum()).reindex(GROUPS)
p10_negative_cost=-f.happiness.clip(upper=0).sum()/len(f)
p10_floor=all_totals.loc['10','mean_happiness']+p10_negative_cost
full_gap_share=p10_negative_cost/(all_totals.loc['6','mean_happiness']-all_totals.loc['10','mean_happiness'])
pd.DataFrame([
    {'scenario':'P10 observed','seats':len(f),'mean_happiness':all_totals.loc['10','mean_happiness']},
    {'scenario':'P10 negatives set to zero','seats':len(f),'mean_happiness':p10_floor},
    {'scenario':'P6 observed','seats':all_totals.loc['6','seats'],'mean_happiness':all_totals.loc['6','mean_happiness']},
]).to_csv(TAB/'full_tournament_downside.csv',index=False)

# Concise report: one explicit claim and one directly supporting figure per item.
TEAL='#087F8C';INK='#17324D';GRAY='#AAB7C2';RED='#BF5346';GOLD='#C49332'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.labelcolor':INK,'text.color':INK,'figure.facecolor':'white','savefig.facecolor':'white'})
def save(name):
    plt.savefig(FIG/f'simple_{name}.png',dpi=210,bbox_inches='tight');plt.close()
def rank_bars(ax,values,label,highlight=None,decimal=2):
    v=values.sort_values(ascending=False)
    ax.barh(range(len(v)),v.values,color=[TEAL if str(g)=='10' else (GOLD if highlight and str(g) in highlight else GRAY) for g in v.index],height=.68)
    ax.set_yticks(range(len(v)),[f'P{g}' for g in v.index]);ax.invert_yaxis();ax.tick_params(axis='y',length=0)
    ax.set_xlim(0,max(v.max()*1.11,1));ax.set_xlabel(label)
    for i,(g,val) in enumerate(v.items()):
        ax.text(val+v.max()*.012,i,f'{val:.{decimal}f}'+('%' if '%' in label else ''),va='center',fontsize=10,fontweight='bold' if str(g)=='10' else 'normal',color=TEAL if str(g)=='10' else INK)
    for t,g in zip(ax.get_yticklabels(),v.index):
        if str(g)=='10':t.set_fontweight('bold');t.set_color(TEAL)
    ax.grid(axis='x',color='#E7EDF1',lw=.6);ax.set_axisbelow(True)

full=standing.assign(group=standing.group.astype(str)).set_index('group')
fig,ax=plt.subplots(figsize=(9,2.25));rank_bars(ax,full.positive_rate*100,'Positive final scores (%) - FULL TOURNAMENT',highlight=['2']);fig.tight_layout();save('00_full_tournament')
fig,ax=plt.subplots(figsize=(9,2.25));rank_bars(ax,equal.positive_rate*100,'Positive final scores (%) - EQUAL-GROUP SUBSET',highlight=['2']);fig.tight_layout();save('01_equal_subset')
fig,ax=plt.subplots(figsize=(9,2.25));a=all_axes['mode'].xs('all').negative_rate*100;rank_bars(ax,a,'Negative final scores (%) - ALL ROSTERS');fig.tight_layout();save('02_losses')

fig,axs=plt.subplots(1,2,figsize=(9,2.55),sharey=True)
for ax,metric,title,mult in [(axs[0],'positive_gap','Positive-score advantage',100),(axs[1],'happiness_gap','Mean-score advantage',1)]:
    v=pair[metric]*mult
    ax.barh(range(9),v,color=[TEAL if x>=0 else RED for x in v]);ax.set_yticks(range(9),[f'vs P{g}' for g in pair.index]);ax.invert_yaxis() if ax==axs[0] else None
    span=max(v.max()-v.min(),1)
    ax.axvline(0,color=INK,lw=.8);ax.set_title(title,fontsize=12);ax.set_xlabel('Percentage points' if mult==100 else 'Happiness points');ax.set_xlim(v.min()-.3*span,v.max()+.3*span);ax.tick_params(axis='y',length=0)
    for i,x in enumerate(v):ax.text(x+(.05 if x>=0 else -.05),i,f'{x:+.2f}',va='center',ha='left' if x>=0 else 'right',fontsize=9)
fig.tight_layout(w_pad=2);save('03_matchups')

dimension_counts=sorted(all_axes['dims'].index.get_level_values('dims').unique())
assert dimension_counts==[1,2,5,8,10,24]
fig,axs=plt.subplots(3,2,figsize=(9,6.6))
for ax,d in zip(axs.flat,dimension_counts):
    values=all_axes['dims'].xs(d).positive_rate*100
    leader=values.drop('10').idxmax()
    rank_bars(ax,values,'Positive scores (%)',highlight=[leader])
    ax.set_xlim(0,112);ax.set_xticks([0,25,50,75,100]);ax.set_title(f'{d} '+('trait' if d==1 else 'traits'),fontsize=12)
fig.tight_layout(w_pad=2,h_pad=1.6);save('04_dimensions')

distributions=[('uniform','Uniform'),('normal','Normal'),('mixed','Mixed'),('right_skew','Right-skewed'),('u_shape','U-shaped'),('p10_90','p10_90'),('p90_10','p90_10')]
assert set(k for k,_ in distributions)==set(all_axes['distribution'].index.get_level_values('distribution'))
fig,axs=plt.subplots(4,2,figsize=(9,8.2))
for ax,(k,label) in zip(axs.flat,distributions):
    values=all_axes['distribution'].xs(k).positive_rate*100
    leader=values.drop('10').idxmax()
    rank_bars(ax,values,'Positive scores (%)',highlight=[leader])
    ax.set_xlim(0,112);ax.set_xticks([0,25,50,75,100]);ax.set_title(label,fontsize=12)
axs.flat[-1].axis('off')
axs.flat[-1].text(.03,.7,'All recorded results\nTeal: Player 10\nGold: strongest other group\n\nSame 0-100% scale in every plot.',transform=axs.flat[-1].transAxes,fontsize=11,color=INK,va='top',linespacing=1.6)
fig.tight_layout(w_pad=2,h_pad=1.6);save('05_distributions')

fig,ax=plt.subplots(figsize=(9,2.25));rank_bars(ax,all_tm.fraction5*100,'Marriages within the first 5% of the horizon (%) - ALL ROSTERS',highlight=['9']);fig.tight_layout();save('06_timing')

fig,ax=plt.subplots(figsize=(9,2.0))
baseline=all_totals.loc['6','mean_happiness']
for i,v in enumerate([all_totals.loc['10','mean_happiness'],p10_floor]):
    ax.hlines(i,v,baseline,color='#D7E1E8',lw=4)
    ax.scatter([v,baseline],[i,i],color=[TEAL,GOLD],s=65,zorder=3)
    ax.annotate(f'P10: {v:.2f}',(v,i),xytext=(0,11),textcoords='offset points',ha='center',color=TEAL,fontweight='bold',fontsize=10)
    ax.annotate(f'P6: {baseline:.2f}',(baseline,i),xytext=(0,11),textcoords='offset points',ha='center',color=GOLD,fontweight='bold',fontsize=10)
    ax.annotate(f'Gap: {baseline-v:.2f}',((v+baseline)/2,i),xytext=(0,-16),textcoords='offset points',ha='center',fontsize=10)
ax.set_yticks([0,1],['Observed','P10 negative scores set to zero']);ax.set_ylim(1.6,-.6);ax.set_xlim(all_totals.loc['10','mean_happiness']-.8,baseline+.8);ax.set_xlabel('Mean happiness (zoomed scale; ALL ROSTERS, including singles)');ax.tick_params(axis='y',length=0)
fig.tight_layout();save('08_downside')

fig,ax=plt.subplots(figsize=(9,2.25));rank_bars(ax,spouse.both_positive_all_seats*100,'Players whose marriage gives BOTH spouses a positive score (%)',highlight=['2','3']);fig.tight_layout();save('09_mutual')

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Claim',fontName='Helvetica-Bold',fontSize=17,leading=20,textColor=colors.HexColor(INK),spaceAfter=7))
styles.add(ParagraphStyle(name='BodyP10',fontSize=10,leading=13.5,spaceAfter=6,textColor=colors.HexColor(INK)))
styles.add(ParagraphStyle(name='NoteP10',fontSize=8,leading=10.5,spaceAfter=7,textColor=colors.HexColor('#53697B')))
styles.add(ParagraphStyle(name='CoverP10',fontName='Helvetica-Bold',fontSize=23,leading=27,textColor=colors.HexColor(INK),spaceAfter=8))
story=[]
def p(t,style='BodyP10'):story.append(Paragraph(t,styles[style]))
def claim(n,t,sample='EQUAL-GROUP SUBSET'):
    p(sample,'NoteP10');p(f'{n}. {t}','Claim')
def fig(n):
    from PIL import Image as PILImage
    im=PILImage.open(FIG/f'simple_{n}.png');story.extend([Image(str(FIG/f'simple_{n}.png'),width=510,height=510*im.height/im.width),Spacer(1,5)])
def gap():story.append(Spacer(1,15))
def page():story.append(PageBreak())

p('Player 10: the clearest findings','CoverP10')
p('Claims 1, 3, 5, 6, 7 and 8 pool ALL roster types. Claims 2 and 9 use equal-group markets; claim 4 uses two-strategy markets. A positive score means happiness above zero; teal highlights Player 10.','NoteP10')
claim(1,'Player 10 finished SEVENTH in the full tournament.','FULL TOURNAMENT')
fig('00_full_tournament')
p('Across all tournament results, Player 10 finishes positive <b>83.15%</b> of the time, behind leader P2 at <b>84.09%</b>. P10 has the highest marriage rate (<b>86.36%</b>), but the standings rank positive outcomes, not marriages.')
p('<b>Interpretation:</b> Our low, falling proposal threshold encourages marriages. Trusting reported profiles may help explain why the highest marriage rate does not translate into the highest positive-score rate.')
p('Full archive: 80,520 runs; 405,540 Player 10 seats. All roster types and settings are included. Different groups receive different mixes of settings and opponents in these pooled standings.','NoteP10')

gap()
claim(2,'Within the equal-group subset, Player 10 ranks second.')
fig('01_equal_subset')
p('Player 10 finishes positive <b>84.22%</b> of the time, versus <b>84.34%</b> for P2: a difference of just <b>18 players out of 14,448</b>. It beats each of the other eight groups in this subset. This does not change its seventh-place full-tournament result.')
p('<b>Interpretation:</b> Our proposal threshold depends on time and bonus, not opponent mix. The ranking change suggests this fixed rule works differently across market compositions; it does not mean the algorithm changed.')
p('Same ten-group markets: 2,394 runs and 14,448 players per group. This is a common-market comparison, not the archive\'s pooled ranking across different market mixes.','NoteP10')

page()
claim(3,'With all traits higher-is-better, Player 10 has the most negative outcomes.','ALL ROSTERS / ALL-MONOTONIC SETTINGS')
fig('02_losses')
p('Its negative-score rate is <b>7.34%</b>. Every other group is between <b>0.23% and 1.26%</b>; even the next-highest, P2, has far fewer losses. P10 also ranks <b>eighth</b> on positive outcomes in these settings, rather than second as it did in the equal-group subset.')
p("<b>Interpretation:</b> We exaggerate higher-is-better traits and trust partners' reports. Combined with a low proposal threshold, this may encourage marriages whose true compatibility is worse than it appears.")
p('Every recorded all-monotonic result, across ALL roster types: 118,332 players per group. A negative outcome means happiness below zero. Opponent and setting mixes may differ by group despite equal total counts.','NoteP10')


gap()
claim(4,'Against most individual opponents, Player 10 gets more positive outcomes but lower average happiness.','TWO-STRATEGY SUBSET')
fig('03_matchups')
p('In two-strategy markets, Player 10 leads <b>six of nine</b> opponents on positive outcomes, but trails <b>eight of nine</b> on mean happiness. <b>P9 is the only opponent it beats on both measures.</b> Teal bars favor P10; red bars favor the named opponent.')
p('<b>Interpretation:</b> Our low threshold and lack of an exploration phase favor early commitment, leaving less opportunity to find higher-value partners. That is consistent with more positive outcomes but lower average scores.')
p('Each row compares both groups in the same half-and-half runs. P1-P8: 31,590 players per side; P9: 54,210 with extra setting coverage. These are separate samples from the ten-group charts.','NoteP10')

page()
claim(5,'Across all results, Player 10 ranks eighth at 10 traits and last at 24.','ALL ROSTERS / DIMENSION BREAKDOWN')
fig('04_dimensions')
p('At 10 traits, P10 finishes positive <b>63.39%</b> of the time and beats only P6 and P4; P2 leads at <b>65.92%</b>. At 24 traits, <b>all nine groups beat P10</b>: its 34.40% trails leader P5\'s 41.20%. The earlier first-place result at 10 traits applied only to equal-group markets.')
p('<b>Interpretation:</b> Our estimates sum squared mismatches across traits but include no uncertainty penalty. With more traits, trusting reported profiles may create larger total estimation errors.')
p('ALL recorded results at every tested dimension count: 1, 2, 5, 8, 10 and 24. Teal: P10; gold: the strongest other group. P10 seats by dimension: 1,872; 124,884; 137,988; 2,808; 135,180; 2,808. Group test coverage varies; pooled ranks do not isolate a causal effect of adding traits.','NoteP10')


page()
claim(6,'Across all results, Player 10 leads on U-shaped data and ranks second on right-skewed data.','ALL ROSTERS / DISTRIBUTION BREAKDOWN')
fig('05_distributions')
p('U-shaped: P10 beats <b>all nine</b> groups at <b>60.79%</b> positive, ahead of P3 at 59.72%. Right-skewed: only P8 beats P10, <b>94.96% to 94.47%</b>. Both rankings differ from the equal-group-only analysis.')
p('<b>Interpretation:</b> Our ad-based squared-gap filter can screen out extreme mismatches before dating; that may help in U-shaped markets. The algorithm does not adapt its rules to the distribution.')
p('ALL recorded results for all seven distributions. P10 seats: uniform/normal 111,780 each; mixed 151,092; each other distribution 7,722. Other groups have different test coverage, so ranks are descriptive. Source labels p10_90 and p90_10 are preserved; teal is P10 and gold the strongest other group.','NoteP10')

page()
claim(7,'Player 10 leads every group in marriages completed within the first 5% of the game.','ALL ROSTERS / MARRIED PLAYERS')
fig('06_timing')
p('<b>66.40%</b> of P10\'s marriages occur by the 5% mark, versus <b>P9\'s 65.49%</b> and P3\'s 49.16%. This early lead is specific to the cutoff: by the 10% mark, P9 is ahead, <b>84.74% versus P10\'s 83.53%</b>.')
p('<b>Interpretation:</b> We skip initial exploration and can propose from the second date. Together with a low threshold that falls over time, these rules make early commitment plausible.')
p('ALL roster types. Denominator: each group\'s MARRIED players, not all players; P10 has 350,208. Timing is marriage round divided by total rounds. Different horizon and market mixes affect pooled timing; it does not show that waiting would help.','NoteP10')


gap()
claim(8,'Erasing negative scores would close only 7.4% of Player 10\'s happiness gap to Player 6.','ALL ROSTERS / ALL PLAYERS')
fig('08_downside')
p(f'Across all results, changing every P10 negative score to zero raises its mean from <b>{all_totals.loc["10","mean_happiness"]:.2f} to {p10_floor:.2f}</b>, still far below P6\'s <b>{baseline:.2f}</b>. The {p10_negative_cost:.3f}-point recovery closes just <b>{100*full_gap_share:.1f}%</b> of the observed gap. The conclusion survives using the full dataset.')
p('<b>Interpretation:</b> Our proposal rule uses a low time-based bar without estimating the value of waiting. It may settle for modest positive matches, making better partner selection more valuable than loss avoidance alone.')
p('ALL roster types: 405,540 P10 players and 428,160 P6 players, including singles. P6 is the mean-happiness leader. This floors negative scores with everything else frozen, not a simulated policy; actual rejections could change future matches. Groups have different test mixes.','NoteP10')


page()
claim(9,'Players 3 and 2 beat Player 10 when both spouses must finish positive.')
fig('09_mutual')
p('P10 ranks first on marriage rate but <b>third</b> when both spouses must score positive: <b>82.14%</b> of all P10 players, behind <b>P3 at 83.01%</b> and <b>P2 at 82.57%</b>. More marriages do not automatically mean more matches that benefit both people.')
p('<b>Interpretation:</b> Our proposal rule checks our estimated score but does not separately require a positive estimated score for the partner. This could explain frequent marriages without leading on mutually positive outcomes.')
p('Same ten-group markets: 14,448 players per group. Each group\'s own players are counted; a same-group couple contributes two players. Partner IDs are joined within each run and checked for reciprocal links.','NoteP10')

gap()
p('How to read the evidence','Claim')
p('Claims 1, 3, 5, 6, 7 and 8 include ALL roster types and every available result relevant to the stated condition. Each numbered group is compared individually using its own observed rate; random players are not a ranked comparator. Opponent and setting mixes differ across groups, so full-results ranks are descriptive, not controlled comparisons.','BodyP10')
p('Only claims 2 and 9 retain the explicitly labeled equal-group subset; claim 4 retains the two-strategy subset. Rates weight players, not runs. Full-results calculations and sample sizes are saved as full_tournament_*.csv alongside the supporting individual tables.','NoteP10')
p('Interpretations describe plausible mechanisms in the checked-out Player 10 algorithm, not proven causes. There are only three seeds; small gaps are not significance tests. The archive has no standard-greedy or previous-version control and does not record player code hashes. Source: tuned_p10_tournament_results.zip. Score, marriage and partner totals are checked. Detailed results remain in output/pdf/player10_tables/individual_*.csv; reproduce with analysis/player10/build_report.py.','NoteP10')

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#D7E1E8'));canvas.line(42,36,570,36)
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor(INK));canvas.drawString(42,23,'PLAYER 10 / TOURNAMENT FINDINGS / 16 NOV 2026');canvas.drawRightString(570,23,str(doc.page))
doc=SimpleDocTemplate(str(OUT/'player10_tournament_report.pdf'),pagesize=(612,792),rightMargin=42,leftMargin=42,topMargin=30,bottomMargin=45,title='Player 10: Clear Tournament Findings',author='Tournament analysis')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print('Saved',OUT/'player10_tournament_report.pdf')
