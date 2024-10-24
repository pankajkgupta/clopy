# %%
'''
**Script to analyse and create CLNF experiment figures.**
Requires clnf_sessions_df.csv
'''

# %%
import os
import platform
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.dirname(parentdir))
sys.path.append("..")
from os.path import dirname, realpath
filepath = realpath(__file__)
dir_of_file = dirname(filepath)
parent_dir_of_file = dirname(dir_of_file)
sys.path.append(parent_dir_of_file)

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
from configparser import ConfigParser
import numpy as np
import tensortools as tt
import seaborn as sns
sns.set(font_scale=0.7)
sns.set_style("ticks")
import helper as clh
import re
from IPython.core.debugger import set_trace
from pytictoc import TicToc
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy import stats,signal,fft
from scipy.stats import f_oneway
from scipy.signal import butter, lfilter, filtfilt
import scipy.io as sio
import scipy
import tables
import h5py
import json
from statannot import add_stat_annotation
import pymannkendall as mk
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
from statannotations.Annotator import Annotator
from pathlib import Path
from numpy import sin, linspace, pi
# from pylab import plot, show, title, xlabel, ylabel, subplot
import ffmpeg
from datetime import datetime
import joypy
import cv2
import io
import imageio
from sklearn import metrics
from sklearn import preprocessing as pp
from numba import jit
from joblib import Parallel, delayed
import multiprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
# import vaex
from sty import fg, bg, ef, rs
from skimage.util import montage
import roi_manager
import argparse
import pickle
import matplotlib
matplotlib.use('Agg')

data_root =  '/home/pankaj/teamshare/pkg/closedloop_fluorescence_data/'
outdir = './processed_data/'
# %%
tmr = TicToc()
dff_response_daily_all = []

df_beh_log_daily_all = pd.DataFrame()
sessions_df = pd.DataFrame(columns=['mouse_id', 'cal_day', 'day', 'sex', 'roi_type', 'roi_rule', 'trials', 'rewards', 'auc', 'target_var', 'exp_label', 'group'])
sessions_df = pd.DataFrame({'mouse_id': pd.Series(dtype='str'),
                            'cal_day': pd.Series(dtype='str'),
                            'day': pd.Series(dtype='str'),
                            'sex': pd.Series(dtype='str'),
                            'roi_type': pd.Series(dtype='str'),
                            'roi_rule': pd.Series(dtype='str'),
                            'trials': pd.Series(dtype='int'),
                            'rewards': pd.Series(dtype='int'),
                            'auc': pd.Series(dtype='float'),
                            'target_var': float,
                            'exp_label': pd.Series(dtype='str'),
                            'group': pd.Series(dtype='str')})
file_dff_response_all_pkl = outdir + os.sep + 'clnf_avg_dff_response_daily' + '.pkl'

file_sessions_df_csv = outdir + os.sep +'clnf_sessions_df.csv'
sessions_df = pd.read_csv(file_sessions_df_csv, dtype = {'mouse_id': str, 'cal_day': str, 'day': int, 'sex': str, 'roi_type': str,
                                                         'roi_rule': str, 'trials': int, 'rewards': int, 'auc': float, 'exp_label': str, 'group': str, 'performance': float})

sessions_df_G1 = sessions_df[sessions_df['group']=='grp1'] #separate G1 G2 dataframes
sessions_df_G2 = sessions_df[sessions_df['group']=='grp2']
day_rulechange = 10
rew_wide_G1 = sessions_df_G1.pivot(index="day", columns=["mouse_id", "group", "sex", 'roi_type', 'roi_rule', "exp_label"], values="rewards") #pivot table of G1 data
rew_wide_G2 = sessions_df_G2.pivot(index="day", columns=["mouse_id", "group", "sex", 'roi_type', 'roi_rule', "exp_label"], values="rewards") #pivot table of G2 data
rew_wide_G2_before = rew_wide_G2.loc[rew_wide_G2.index <= day_rulechange, :] #separate G2 pivot table in before after tables
rew_wide_G2_after = rew_wide_G2.loc[rew_wide_G2.index > day_rulechange, :]
rew_wide_G1 = (rew_wide_G1-rew_wide_G1.min())/(rew_wide_G1.max()-rew_wide_G1.min()) #normalize
rew_wide_G2_before = (rew_wide_G2_before-rew_wide_G2_before.min())/(rew_wide_G2_before.max()-rew_wide_G2_before.min()) #normalize
rew_wide_G2_after = (rew_wide_G2_after-rew_wide_G2_after.min())/(rew_wide_G2_after.max()-rew_wide_G2_after.min()) # normalize
rew_wide = pd.concat([rew_wide_G1, rew_wide_G2_before, rew_wide_G2_after]); #combine after normalization
rew_wide = rew_wide.reset_index() #reset index for us to index 'day'
rew_long = rew_wide.melt('day', var_name=['mouse_id','group', 'sex', 'roi_type', 'roi_rule', 'exp_label'], value_name='rewards') # convert to long form for plotting

#%%
pp = PdfPages(outdir + os.path.basename(__file__).split('.')[0] + '_' + 'summary_stats.pdf')

#%% KDE plot for figure 1
fig = plt.figure(figsize=(6, 3))
sns.kdeplot(data=df_beh_log_daily_all.loc[(df_beh_log_daily_all['day'].isin([1,9])) & (df_beh_log_daily_all['exp_label'] == 'exp10')], x='target_dff',
            hue='day', log_scale=(False, False), shade=True, palette='Set1') #husl, colorblind, Set1, bright
plt.xlim([-0.2, 0.2])
sns.despine(offset=10, trim=True)
plt.title(' group DFF distribution whole sessions')
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.xlabel('dff', fontweight='bold', fontsize=14)
plt.ylabel('density', fontweight='bold', fontsize=14)
plt.tight_layout(pad=2);
pp.savefig(fig); plt.close()

#%% avg reward centered dff response plot for figure 1
brain_fps = 30
epochSize = int(3*brain_fps)
rew_ixs = df_beh_log_daily_all.loc[(df_beh_log_daily_all['day'].isin([1,9])) &
                                   (df_beh_log_daily_all['exp_label'] == 'exp14') &
                                   (df_beh_log_daily_all['reward'] == 1)].index
rew_ixs = rew_ixs[1:][np.diff(rew_ixs) > 2 * epochSize]
tw = 2 * int(brain_fps)
t = np.arange(-tw, 0) / brain_fps
df = pd.DataFrame();
df = df.append([df_beh_log_daily_all.iloc[ix - tw:ix].reset_index() for ix in rew_ixs])
df.reset_index(inplace=True)
fig = plt.figure(figsize=(6, 3))
gfg = sns.lineplot(data=df, x='level_0', y="target_dff", hue='day', palette='Set1') #husl, colorblind, Set1, bright
sns.despine(offset=10, trim=True)
plt.xlabel('Time (before reward)', fontweight='bold', fontsize=14)
plt.xticks(np.arange(0, len(t), 15), t[::15], fontsize=14)
plt.yticks(fontsize=14);
plt.tight_layout(pad=2);
pp.savefig(fig); plt.close()

#%% block to plot and compare ROI1 and ROI2 data in 2ROI expts
unique_expts_2roi = df_beh_log_daily_all[df_beh_log_daily_all['roi_type']=='2ROI'][['mouse_id', 'exp_label', 'roi_type']].drop_duplicates().reset_index(drop=True)
for i, e in unique_expts_2roi.iterrows():
    df = df_beh_log_daily_all[(df_beh_log_daily_all['mouse_id'] == e[0]) &
                              (df_beh_log_daily_all['exp_label'] == e[1]) &
                              (df_beh_log_daily_all['roi_type'] == '2ROI')]
    
    g = sns.FacetGrid(df, col="day")
    g.map(sns.regplot, 'roi1dff', 'roi2dff', scatter_kws={"color": "forestgreen", 's':0.5}, line_kws={"color": "darkviolet"})
    
    plt.tight_layout()
    plt.title(e[0] + ' ' + e[1])
    plt.tight_layout();
    pp.savefig(plt.gcf()); plt.close()

#%% Figure 4G jointplot of ROI1 and ROI2 dff in 2ROI data
df = df_beh_log_daily_all[(df_beh_log_daily_all['mouse_id'] == 'BM1') &
                          (df_beh_log_daily_all['exp_label'] == 'exp16') &
                          (df_beh_log_daily_all['day'].isin([6,17]))]
sns.jointplot(data=df, x="roi1dff", y="roi2dff", hue="day", kind="kde", n_levels=5, palette='Set2', fill=True, alpha=0.5)
pp.savefig(plt.gcf()); plt.close()

###### get slope values using regplot for figure 4G jointplot
# plt.figure();
# df = df_beh_log_daily_all[(df_beh_log_daily_all['mouse_id'] == 'BM1') &
#                           (df_beh_log_daily_all['exp_label'] == 'exp16') &
#                           (df_beh_log_daily_all['day'].isin([6]))]
# g = sns.regplot(data=df, x="roi1dff", y="roi2dff"); plt.xlim([-0.2, 0.2]); plt.ylim([-0.2, 0.2])
# slope, intercept, r, p, sterr = scipy.stats.linregress(x=g.get_lines()[0].get_xdata(), y=g.get_lines()[0].get_ydata())
# plt.text(0, 0, 'y = ' + str(round(intercept,6)) + ' + ' + str(round(slope,6)) + 'x')
##### get stats for bivariate distribution comparison: Multivariate Two-Sample Test using Permutation Test
# data11 = df_beh_log_daily_all[(df_beh_log_daily_all['mouse_id'] == 'BM1') &
#                           (df_beh_log_daily_all['exp_label'] == 'exp16') &
#                           (df_beh_log_daily_all['day'].isin([6]))][['roi2dff', 'roi2dff']].values
# data2 = df_beh_log_daily_all[(df_beh_log_daily_all['mouse_id'] == 'BM1') &
#                           (df_beh_log_daily_all['exp_label'] == 'exp16') &
#                           (df_beh_log_daily_all['day'].isin([17]))][['roi2dff', 'roi2dff']].values
# from hyppo.ksample import MMD
# stat, p_val = MMD(compute_kernel="rbf").test(data1, data2)

#%% Figure 5C regplot of ROI1 and ROI2 dff
df = df_beh_log_daily_all[(df_beh_log_daily_all['mouse_id'] == 'BM1') &
                          (df_beh_log_daily_all['exp_label'] == 'exp16') &
                          (df_beh_log_daily_all['roi_type'] == '2ROI')]
g = sns.FacetGrid(df, col="day")
g.map(sns.regplot, 'roi1dff', 'roi2dff', scatter_kws={"color": "turquoise", 's':0.5}, line_kws={"color": "magenta"})
pp.savefig(plt.gcf()); plt.close()

#%% stats for paper
fig = plt.figure(figsize=(12,10))
fig.clf()
fig.text(0.05,0.97, 'Mann-Kendall test on group performance before rule change', weight='bold')
fig.text(0.05,0.95, mk.original_test(rew_wide.iloc[:,0:].mean(axis=1)[:day_rulechange]))
fig.text(0.05,0.93, 'Mann-Kendall test on group performance after rule change', weight='bold')
fig.text(0.05,0.91, mk.original_test(rew_wide.iloc[:,0:].mean(axis=1)[day_rulechange:]))
### repeated measure ANOVA - https://datatab.net/tutorial/anova-with-repeated-measures
a = rew_long[(rew_long['day']<day_rulechange) & (rew_long['group']=='grp1')]
fig.text(0.05,0.89, 'RM-ANOVA G1 before rule change', weight='bold')
fig.text(0.05,0.82, pg.rm_anova(dv='rewards', within=['day'], subject='mouse_id', data=a.fillna(0), detailed=True).to_markdown())
### RM ANOVA using statsmodels package
# print(AnovaRM(data=a.fillna(0), depvar='rewards', subject='mouse_id', within=['day']).fit())
# optional post-hoc test
# post_hoc = pg.pairwise_ttests(dv='rewards', within=['day'], subject='mouse_id', padjust='fdr_bh', data=a.fillna(0))
### RM ANOVA on data before the rule change to see the effect of day on perfromance
a = rew_long[(rew_long['day']>day_rulechange) & (rew_long['group']=='grp2')]
fig.text(0.05,0.78, 'RM-ANOVA G2 after rule change', weight='bold')
fig.text(0.05,0.72, pg.rm_anova(dv='rewards', within=['day'], subject='mouse_id', data=a.fillna(0), detailed=True).to_markdown())
### Two-way RM ANOVA on day and group variables, before the rule change to see if there was an effect of day and group on their performance
a = rew_long[(rew_long['day']<day_rulechange)]
fig.text(0.05,0.69, 'Two-way RM-ANOVA G1 & G2 before rule change', weight='bold')
fig.text(0.05,0.62, pg.rm_anova(dv='rewards', within=['day','group'], subject='mouse_id', data=a.fillna(0), detailed=True).to_markdown())
### one-way ANOVA bw day10 and day11 of G2 to test significance
rewards_G2_D10 = rew_long[(rew_long['day']==10) & (rew_long['group']=='grp2')].dropna().rewards.values
rewards_G2_D11 = rew_long[(rew_long['day']==11) & (rew_long['group']=='grp2')].dropna().rewards.values
res = f_oneway(rewards_G2_D10, rewards_G2_D11)
fig.text(0.05,0.58, 'one-way ANOVA bw day10 and day11 of G2 to test significance of rule change within group', weight='bold')
fig.text(0.05,0.54, res)
### one-way ANOVA day11 bw G1 and G2 to test significance
rewards_G1_D11 = rew_long[(rew_long['day']==11) & (rew_long['group']=='grp1')].dropna().rewards.values
rewards_G2_D11 = rew_long[(rew_long['day']==11) & (rew_long['group']=='grp2')].dropna().rewards.values
res = f_oneway(rewards_G1_D11, rewards_G2_D11)
fig.text(0.05,0.49, 'one-way ANOVA bw day11 G1 and day11 of G2 to test significance of rule change between the groups', weight='bold')
fig.text(0.05,0.45, res)

pp.savefig(fig); plt.close()
#%% Figure 4A CLNF success rate by group
hue_plot_params = {
    'data': rew_long,
    'x': 'day',
    'y': 'rewards',
    "hue": "group",
    "palette": 'husl'
}
fig, ax = plt.subplots(figsize=(12, 8))
gfg = sns.pointplot(**hue_plot_params, ax=ax, dodge=0.2, linewidth=1)
# gfg = sns.pointplot(x='day', y="rewards", hue='group', data=rew_long, kind='point', dodge=True, linewidth=1, markersize=5, palette="husl"); plt.show()
# gfg = sns.lineplot(x='day', y="rewards", hue='group', data=rew_long, palette="husl", err_style="bars", marker='o', markersize=5, linewidth=1);
gfg.legend(fontsize=15)
sns.despine(offset=10, trim=True)
plt.xlabel('Days', fontweight='bold', fontsize=18)
plt.xticks(rew_long.day.unique()-1, rew_long.day.unique(), fontsize=20)
plt.ylabel('Success Rate', fontweight='bold', fontsize=18)
plt.yticks(fontsize=20)
plt.title('Success rate G1 G2', fontsize=18)
pairs = [
    [(1, 'grp1'), (3, 'grp1')],
    [(10, 'grp1'), (10, 'grp2')],
    [(10, 'grp2'), (11, 'grp2')],
    [(10, 'grp2'), (19, 'grp2')],
]
# annotator = Annotator(ax, pairs, data=successrate_by_mouseid, x='day', y='reward')
annotator = Annotator(gfg, pairs, **hue_plot_params)
# # test: Brunner-Munzel, Levene, Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, t-test_ind, t-test_welch, t-test_paired, Wilcoxon, Kruskal
# # comparisons_correction: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg, Benjamini-Yekutieli
annotator.configure(test='Mann-Whitney', show_test_name=True, comparisons_correction='Benjamini-Hochberg',
                    text_format='star')  # bejamini hochberg, Bonferroni
annotator.apply_and_annotate()
pp.savefig(plt.gcf())
plt.close();
#%% Supplementary figure 3A, male female success rate
hue_plot_params = {
    'data': rew_long,
    'x': 'day',
    'y': 'rewards',
    "hue": "sex",
    "palette": 'husl'
}
fig, ax = plt.subplots(figsize=(12, 8))
gfg = sns.pointplot(**hue_plot_params, ax=ax, dodge=0.2, linewidth=1)
# gfg = sns.pointplot(x='day', y="rewards", hue='group', data=rew_long, kind='point', dodge=True, linewidth=1, markersize=5, palette="husl"); plt.show()
# gfg = sns.lineplot(x='day', y="rewards", hue='group', data=rew_long, palette="husl", err_style="bars", marker='o', markersize=5, linewidth=1);
gfg.legend(fontsize=15)
sns.despine(offset=10, trim=True)
plt.xlabel('Days', fontweight='bold', fontsize=18)
plt.xticks(rew_long.day.unique()-1, rew_long.day.unique(), fontsize=20)
plt.ylabel('Success Rate', fontweight='bold', fontsize=18)
plt.yticks(fontsize=20)
plt.title('Success rate male female', fontsize=18)
pairs = [
    [(1, 'male'), (3, 'male')],
    [(3, 'male'), (3, 'female')],
    [(4, 'male'), (4, 'female')],
    [(10, 'male'), (10, 'female')],
    [(10, 'female'), (11, 'female')],
]
# annotator = Annotator(ax, pairs, data=successrate_by_mouseid, x='day', y='reward')
annotator = Annotator(gfg, pairs, **hue_plot_params)
# # test: Brunner-Munzel, Levene, Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, t-test_ind, t-test_welch, t-test_paired, Wilcoxon, Kruskal
# # comparisons_correction: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg, Benjamini-Yekutieli
annotator.configure(test='Mann-Whitney', show_test_name=True, comparisons_correction='Benjamini-Hochberg',
                    text_format='star')  # bejamini hochberg, Bonferroni
annotator.apply_and_annotate()
pp.savefig(plt.gcf())
plt.close();
#%% Supplementary figure 3B, 1ROI vs 2ROI success rate
hue_plot_params = {
    'data': rew_long,
    'x': 'day',
    'y': 'rewards',
    "hue": "roi_type",
    "palette": 'husl'
}
fig, ax = plt.subplots(figsize=(12, 8))
gfg = sns.pointplot(**hue_plot_params, ax=ax, dodge=0.2, linewidth=1)
# gfg = sns.pointplot(x='day', y="rewards", hue='group', data=rew_long, kind='point', dodge=True, linewidth=1, markersize=5, palette="husl"); plt.show()
# gfg = sns.lineplot(x='day', y="rewards", hue='group', data=rew_long, palette="husl", err_style="bars", marker='o', markersize=5, linewidth=1);
gfg.legend(fontsize=15)
sns.despine(offset=10, trim=True)
plt.xlabel('Days', fontweight='bold', fontsize=18)
plt.xticks(rew_long.day.unique()-1, rew_long.day.unique(), fontsize=20)
plt.ylabel('Success Rate', fontweight='bold', fontsize=18)
plt.yticks(fontsize=20)
plt.title('Success rate 1ROI vs 2ROI', fontsize=18)
pairs = [
    [(1, '1ROI'), (3, '1ROI')],
    [(3, '1ROI'), (3, '2ROI')],
    [(4, '1ROI'), (4, '2ROI')],
    [(10, '1ROI'), (10, '2ROI')],
    [(10, '2ROI'), (11, '2ROI')],
]
# annotator = Annotator(ax, pairs, data=successrate_by_mouseid, x='day', y='reward')
annotator = Annotator(gfg, pairs, **hue_plot_params)
# # test: Brunner-Munzel, Levene, Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, t-test_ind, t-test_welch, t-test_paired, Wilcoxon, Kruskal
# # comparisons_correction: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg, Benjamini-Yekutieli
annotator.configure(test='Mann-Whitney', show_test_name=True, comparisons_correction='Benjamini-Hochberg',
                    text_format='star')  # bejamini hochberg, Bonferroni
annotator.apply_and_annotate()
pp.savefig(plt.gcf())
plt.close();

#%% Supplementary figure 3D success rate fast learner roi_rules vs slow learning roi_rules
rew_long_d1d10 = rew_long[rew_long['day'].isin([1,2,3,4,5,6,7,8,9,10])]
roi_rule_list = rew_long_d1d10.roi_rule.unique()

a1 = rew_long_d1d10[rew_long_d1d10['roi_rule'].isin(['RL_R','M2_L','M1_R','RL_L','roi2-roi1'])]
a2 = rew_long_d1d10[rew_long_d1d10['roi_rule'].isin(['M1_L', 'V1_L','BC_L-BC_R','BC_L-HL_L','HL_L-HL_R','roi1-roi2'])]
qual_palette = sns.color_palette("Reds", len(roi_rule_list))

fig, ax = plt.subplots(figsize=(12, 8))
gfg = sns.pointplot(data=a1, x='day', y='rewards', ax=ax, color='b')
gfg = sns.pointplot(data=a2, x='day', y='rewards', ax=ax, color='r')
gfg.legend(fontsize=15)
sns.despine(offset=10, trim=True)
plt.xlabel('Days', fontweight='bold', fontsize=18)
plt.ylabel('Success Rate', fontweight='bold', fontsize=18)
plt.xticks(rew_long_d1d10.day.unique()-1, rew_long_d1d10.day.unique(), fontsize=20)
plt.yticks(fontsize=20)
plt.title('Success rate by ROI rule', fontsize=18)

#%% Supplementary figure 3 - not included in paper but helpful in exploration, success rate by roi-rule
######## visualize success rate per roi rule
for roi in rew_long_d1d10.roi_rule.unique():
    a1 = rew_long_d1d10[rew_long_d1d10.roi_rule==roi]
    if a1.rewards.any():
        fig, ax = plt.subplots(figsize=(12, 8))
        gfg = sns.pointplot(data=a1, x='day', y='rewards', ax=ax, legend=True)
        plt.legend(fontsize=15)
        sns.despine(offset=10, trim=True)
        plt.xlabel('Days', fontweight='bold', fontsize=18)
        plt.xticks(rew_long_d1d10.day.unique(), rew_long_d1d10.day.unique(), fontsize=20)
        plt.ylabel('Success Rate', fontweight='bold', fontsize=18)
        plt.yticks(fontsize=20)
        plt.title('Success rate by ROI rule '+roi, fontsize=18)
        print(scipy.stats.linregress(x=a1.day.values, y=a1.rewards.values))
        
#%%
regslope_df = []
for roi in rew_long_d1d10.roi_rule.unique():
    a1 = rew_long_d1d10[rew_long_d1d10.roi_rule==roi]
    if a1.rewards.any():
        fig, ax = plt.subplots(figsize=(12, 8))
        g = sns.regplot(data=a1, x="day", y="rewards", ax=ax);
        sns.despine(offset=10, trim=True)
        plt.xlabel('Days', fontweight='bold', fontsize=18)
        plt.xticks(rew_long_d1d10.day.unique(), rew_long_d1d10.day.unique(), fontsize=20)
        plt.ylabel('Success Rate', fontweight='bold', fontsize=18)
        plt.yticks(fontsize=20)
        plt.title('Success rate by ROI rule '+roi, fontsize=18)
        
        slope, intercept, r, p, sterr = scipy.stats.linregress(x=g.get_lines()[0].get_xdata(), y=g.get_lines()[0].get_ydata())
        plt.text(0.5, 0, 'y = ' + str(round(intercept,6)) + ' + ' + str(round(slope,6)) + 'x')
        print(roi, '\t',slope, '\t', intercept, '\t', r, '\t', p, '\t', sterr)
        
        d = {'roi' : roi, 'slope' : slope, 'intercept' : intercept, 'r' : r, 'p' : p, 'sterr' : sterr }
        regslope_df.append(d)
regslope_df = pd.DataFrame(regslope_df)

sns.barplot(data=regslope_df, x="roi", y="slope", errorbar="sd", color='grey');
plt.ylim([0, 0.2]); 
sns.despine(offset=10, trim=True)
plt.xticks(rotation=60);
plt.tight_layout()

#%% Supplementary figure 3C: helps determine slope of overall data (mean slope)
g = sns.regplot(data=rew_long_d1d10, x="day", y="rewards");
sns.despine(offset=10, trim=True)
plt.xlabel('Days', fontweight='bold', fontsize=18)
plt.xticks(rew_long_d1d10.day.unique()-1, rew_long_d1d10.day.unique(), fontsize=20)
plt.ylabel('Success Rate', fontweight='bold', fontsize=18)
plt.yticks(fontsize=20)
slope, intercept, r, p, sterr = scipy.stats.linregress(x=g.get_lines()[0].get_xdata(), y=g.get_lines()[0].get_ydata())
#%%
a = sessions_df.melt(id_vars=['cal_day','day'], value_vars=['rewards'])
fig = plt.figure(figsize=(8, 5));
# gfg = sns.boxplot(x="day", y="value", hue="variable", data=a)
gfg = sns.pointplot(x="day", y="value", hue="variable", data=a, errorbar=('ci', 95), dodge=True, linewidth=2)
gfg.legend(fontsize=15); sns.despine(offset=10, trim=True)
pp.savefig(fig)
plt.close()

pp.savefig(fig)

pp.close()