"""
Created on Thu Jul 25 12:09:03 2024
use trials_df and plot data stats and performance
Requires file clmf_df_speed_daily_all.pkl clmf_df_beh_log_daily_all.pkl, clmf_avg_dff_response_daily.pkl
Please download from here-https://drive.google.com/drive/folders/1eq0muLtc36jcU4sUy8fVOSmN4dxVxdn8?usp=sharing
@author: pankaj gupta
"""
import os
import platform
import sys
from os.path import dirname, realpath
filepath = realpath(__file__)
dir_of_file = dirname(filepath)
parent_dir_of_file = dirname(dir_of_file)
sys.path.append(parent_dir_of_file)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors
import numpy as np
from statannot import add_stat_annotation
import pymannkendall as mk
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
from scipy.stats import f_oneway
from statannotations.Annotator import Annotator
import itertools
import tifffile as tif
import pandas as pd
import seaborn as sns
from pathlib import Path
import glob
from tqdm import tqdm
from scipy import signal
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import imageio
import argparse
from configparser import ConfigParser
import napari
import pickle
os.environ[ 'NUMBA_CACHE_DIR' ] = '/tmp/numba_cache/'
import dlc2kinematics
from statsmodels.stats.multitest import multipletests
# from pygifsicle import optimize
from get_clmf_data import \
    expt1_data_list, expt2_data_list, expt3_data_list
from helper import *

data_root = '/home/pankaj/teamshare/pkg/closedloop_rig5_data/'
out_dir = '../processed_data/'

data_list = expt1_data_list
data_list.extend(expt2_data_list)
data_list.extend(expt3_data_list)

# %%
# Initiate the parser
parser = argparse.ArgumentParser()
# Add long and short argument
parser.add_argument("--process_mouse", "-n", required=False, help="Folder name to process")
parser.add_argument("--process_dir", "-d", type=int, required=False, help="Set recording dir to process")
# Read arguments from the command line
args = parser.parse_args()
if args.process_mouse: # Check for --name
    print("Process mouse id %s" % args.process_mouse)
if args.process_dir: # Check for --day
    print("Process dir %s" % args.process_dir)

ppmm_brain = 14.56
ppmm_beh = 3.4

mice_list = [e[0] for e in data_list]
groups = [e[2] for e in data_list]
cal_days = list(itertools.chain.from_iterable([e[1] for e in data_list]))

pp = PdfPages(out_dir + os.path.basename(__file__).split('.')[0] + '_summary_stats.pdf')
#%%
file_sessions_df_csv = out_dir + os.sep + 'clmf_sessions_df' + '.csv'
if os.path.isfile(file_sessions_df_csv):
    sessions_df = pd.read_csv(file_sessions_df_csv, dtype = {'mouse_id':str, 'cal_day':str, 'day':int, 'control_joint':str,
                                                             'trials':int, 'rewards':int, 'fll_based_rewards':int, 
                                                             'flr_based_rewards':int, 'fll_totaldist': float, 'flr_totaldist': float, 
                                                             'pca_dims_brain': float, 'pca_dims_behavior': float, 'group':str})
    
    sessions_df = sessions_df[sessions_df['mouse_id'].isin( mice_list)]
    sessions_df = sessions_df[sessions_df['group'].isin( groups)]
    sessions_df = sessions_df[sessions_df['cal_day'].isin(cal_days)]
    #%% Plot left and right paw movements over days
    for grp in np.unique(groups):
        print(grp)
        a = sessions_df[sessions_df['group'].isin([grp])]
        sessions_df_melted = a.melt(id_vars=['cal_day','day'], value_vars=['fll_totaldist', 'flr_totaldist'])
        fig = plt.figure();
        # gfg = sns.boxplot(x="day", y="value", hue="variable", data=sessions_df_melted)
        gfg = sns.pointplot(x="day", y="value", hue="variable", data=sessions_df_melted, errorbar=('ci', 95), dodge=True)
        gfg.legend(fontsize=15)
        sns.despine(offset=10, trim=True)
        plt.xlabel('Days', fontweight='bold', fontsize=14)
        plt.xticks(sessions_df.day.unique(), sessions_df.day.unique() +1, fontsize=20)
        plt.ylabel('Paw movement (mm)', fontweight='bold', fontsize=14)
        plt.yticks(fontsize=20)
        plt.title(grp + ' paw movements', fontsize=14)
        plt.tight_layout();
        pp.savefig(fig); plt.close()
    
    #%% Left and right paw movement based rewards over days
    for grp in np.unique(groups):
        print(grp)
        a = sessions_df[sessions_df['group'].isin([grp])]
        sessions_df_melted = a.melt(id_vars=['cal_day','day'], value_vars=['fll_based_rewards', 'flr_based_rewards'])
        sessions_df_melted['value'] = sessions_df_melted['value'].div(60).mul(100)
        fig = plt.figure();
        # gfg = sns.boxplot(x="day", y="value", hue="variable", data=sessions_df_melted)
        # gfg = sns.lineplot(x="day", y="value", hue="variable", data=sessions_df_melted, err_style="bars", alpha=0.5)
        gfg = sns.pointplot(x="day", y="value", hue="variable", data=sessions_df_melted, errorbar=('ci', 95), dodge=True)
        gfg.legend(fontsize=15)
        sns.despine(offset=10, trim=True)
        plt.xlabel('Days', fontweight='bold', fontsize=14)
        plt.xticks(sessions_df.day.unique(), sessions_df.day.unique() +1, fontsize=20)
        plt.ylabel('Success Rate (%)', fontweight='bold', fontsize=14)
        plt.yticks(fontsize=20)
        plt.title(grp + ' FLL and FLR based rewards', fontsize=14);
        plt.tight_layout();
        pp.savefig(fig); plt.close()
    
    #%% Plot success rate over days for each group
    sessions_df['performance'] = (sessions_df['rewards'] - sessions_df['rewards'].min()) / (sessions_df['rewards'].max() - sessions_df['rewards'].min())
    fig = plt.figure(figsize=(12,8));
    gfg = sns.pointplot(x="day", y="performance", hue="group", data=sessions_df, dodge=0.2)
    gfg.legend(fontsize=15)
    sns.despine(offset=10, trim=True)
    plt.xlabel('Days', fontweight='bold', fontsize=20)
    plt.xticks(sessions_df.day.unique(), sessions_df.day.unique() +1, fontsize=20)
    plt.ylabel('Normalized', fontweight='bold', fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Group success rate', fontsize=20)
    plt.tight_layout();
    pp.savefig(fig); plt.close()
    
    #%% Write statistical results summary in the pdf
    fig = plt.figure(figsize=(12,10))
    fig.clf()
    
    a = sessions_df[(sessions_df['day']<=4) & (sessions_df['group']=='grp1')]
    fig.text(0.05,0.97, 'RM-ANOVA grp1 before rule change', weight='bold')
    fig.text(0.05,0.89, pg.rm_anova(dv='performance', within=['day'], subject='mouse_id', data=a, detailed=True).to_markdown())
    
    a = sessions_df[(sessions_df['day']>4) & (sessions_df['group']=='grp1')]
    fig.text(0.05,0.81, 'RM-ANOVA grp1 after rule change', weight='bold')
    fig.text(0.05,0.73, pg.rm_anova(dv='performance', within=['day'], subject='mouse_id', data=a, detailed=True).to_markdown())
    
    a = sessions_df[(sessions_df['group']=='grp1')]
    fig.text(0.05,0.65, 'RM-ANOVA grp1 on all days', weight='bold')
    fig.text(0.05,0.57, pg.rm_anova(dv='performance', within=['day'], subject='mouse_id', data=a, detailed=True).to_markdown())
    
    a = sessions_df[(sessions_df['group']=='grp2')]
    fig.text(0.05,0.49, 'RM-ANOVA grp2 on all days', weight='bold')
    fig.text(0.05,0.41, pg.rm_anova(dv='performance', within=['day'], subject='mouse_id', data=a, detailed=True).to_markdown())
    
    a = sessions_df[(sessions_df['group']=='grp3')]
    fig.text(0.05,0.33, 'RM-ANOVA grp3 on all days', weight='bold')
    fig.text(0.05,0.25, pg.rm_anova(dv='performance', within=['day'], subject='mouse_id', data=a, detailed=True).to_markdown())
    
    pp.savefig(fig)
    plt.close()
    
#%%
file_trials_df_csv = out_dir + os.sep + 'clmf_trials_df' + '.csv'
if os.path.isfile(file_trials_df_csv):

    trials_df = pd.read_csv(file_trials_df_csv, dtype = {'mouse_id':str, 'cal_day':str, 'day':int, 'control_joint':str, 'group':str,
                                                         'trial':int, 'start_ix':int, 'end_ix':int, 'tr_duration':float,
                                                         'reward': int, 'manual_label': str, 
                                                         'fll_dist_tr': float, 'fll_dist_prestart': float, 'fll_dist_poststart': float,
                                                         'fll_dist_prereward': float, 'fll_dist_postreward': float, 'fll_maxspeed_prereward': float,
                                                         'flr_dist_tr': float, 'flr_dist_prestart': float, 'flr_dist_poststart': float, 
                                                         'flr_dist_prereward': float, 'flr_dist_postreward': float, 'flr_maxspeed_prereward': float,
                                                         'fll_tortuosity_prereward': float, 'flr_tortuosity_prereward': float})

    # we can select part of the dataframe based on current analysis
    trials_df = trials_df[trials_df['mouse_id'].isin( mice_list)]
    trials_df = trials_df[trials_df['group'].isin( groups)]
    trials_df = trials_df[trials_df['cal_day'].isin(cal_days)]
    
    #%% re-assign 'day' column in sequential  order based on datasets includeddd
    for expt in data_list:
        mouse_id, list_rec_dir, grp = expt        
        for day, rec_dir in enumerate(list_rec_dir):
            if ((trials_df['mouse_id'] == mouse_id) & (trials_df['cal_day'] == rec_dir)).any():
                #row already exists, update all values except the manual_label
                trials_df.loc[(trials_df['mouse_id'] == mouse_id) & (trials_df['cal_day'] == rec_dir), ['day']] = day
    
    #%%
    for grp in np.unique(groups):
        print(grp)
        a = trials_df[trials_df['group'].isin([grp])]
        trials_df_melted = a.melt(id_vars=['cal_day','day'], value_vars=['fll_maxspeed_prereward', 'flr_maxspeed_prereward'])
        fig = plt.figure(figsize=(8,5));
        # gfg = sns.boxplot(x="day", y="value", hue="variable", data=trials_df_melted)
        gfg = sns.pointplot(x="day", y="value", hue="variable", data=trials_df_melted, errorbar=('ci', 95), dodge=True) #husl, colorblind, Set1, bright 
        gfg.legend(fontsize=15)
        sns.despine(offset=10, trim=True)
        # plt.ylim([0,120])
        plt.xlabel('Days', fontweight='bold', fontsize=18)
        plt.xticks(np.arange(len(trials_df.day.unique())), trials_df.day.unique(), fontsize=20)
        plt.ylabel('Speed (mm/sec.)', fontweight='bold', fontsize=18)
        plt.yticks(fontsize=20)
        plt.title(grp + ': Maximum L/R paw speed during trials', fontsize=14)
        plt.tight_layout();
        pp.savefig(fig); plt.close()
    
    #%%
    trialcounts_by_mouseid_all = trials_df.groupby(['mouse_id', 'day', 'group'])['reward'].count()
    trialcounts_by_mouseid_rewarded = trials_df[trials_df.reward==1].groupby(['mouse_id', 'day', 'group'])['reward'].count()
    successrate_by_mouseid = (trialcounts_by_mouseid_rewarded / trialcounts_by_mouseid_all).reset_index()
    hue_plot_params = {
        'data': successrate_by_mouseid,
        'x': 'day',
        'y': 'reward',
        "hue": "group",
        "palette": 'husl'
    }
    fig = plt.figure(); 
    ax = sns.pointplot(**hue_plot_params, dodge=0.2)
    sns.despine(offset=10, trim=True)
    plt.xlabel('Days', fontweight='bold', fontsize=14)
    plt.xticks(trials_df.day.unique(), trials_df.day.unique() +1, fontsize=20)
    plt.ylabel('Success Rate (%)', fontweight='bold', fontsize=14)
    plt.yticks(fontsize=20)
    plt.title('Group success rate')
    plt.tight_layout(pad=2)
    pairs = [
    [(1, 'grp1'), (3, 'grp1')],
    [(1, 'grp1'), (4, 'grp1')],
    [(1, 'grp1'), (5, 'grp1')],
    [(4, 'grp1'), (5, 'grp1')],
    [(5, 'grp1'), (6, 'grp1')],
    [(1, 'grp3'), (4, 'grp3')],
    ]
    # annotator = Annotator(ax, pairs, data=successrate_by_mouseid, x='day', y='reward')
    annotator = Annotator(ax, pairs, **hue_plot_params)
    # # test: Brunner-Munzel, Levene, Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, t-test_ind, t-test_welch, t-test_paired, Wilcoxon, Kruskal
    # # comparisons_correction: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg, Benjamini-Yekutieli
    annotator.configure(test='Mann-Whitney', show_test_name=True, comparisons_correction='Benjamini-Hochberg', text_format='star') #bejamini hochberg, Bonferroni
    annotator.apply_and_annotate()
    pp.savefig(fig); plt.close()
    
    #%%
    fig = plt.figure(figsize=(8,5))
    # ax = sns.violinplot(x='day', y='tr_duration', hue='group', data=trials_df,inner="box",  palette="husl", linewidth=0.5, saturation=0.5)
    # ax = sns.lineplot(x='day', y='tr_duration', hue='group', data=trials_df, errorbar=('ci', 95), err_style="bars", markers = True, linewidth=2, alpha=0.5)
    ax = sns.pointplot(x='day', y='tr_duration', hue='group', data=trials_df, errorbar=('ci', 95))
    sns.despine(offset=10, trim=True)
    plt.xlabel('Day', fontweight='bold', fontsize=14)
    plt.xticks(trials_df.day.unique(), trials_df.day.unique() +1, fontsize=20)
    # plt.axvline(x=5, c='k')
    plt.ylabel('Task delay (s)', fontweight='bold', fontsize=14)
    plt.yticks(fontsize=20)
    plt.title('Group task delay', fontsize=14)
    plt.tight_layout();
    # pairs=[(0,3), (0,4), (3,4), (4,5),(4,9),(8,9)]
    # annotator = Annotator(ax, pairs, data=trials_df, x='day', y='tr_duration')
    # annotator.configure(test='Mann-Whitney', show_test_name=True, comparisons_correction='Benjamini-Hochberg', text_format='star')
    # annotator.apply_and_annotate()
    pp.savefig(fig)
    plt.close()
    print('Done')

#%% Requires kld_df.csv located under data directory
bp_select = ['snout_top', 'fl_l', 'fl_r', 'hl_l', 'hl_r', 'tailbase_bottom']
file_kld_df_csv = out_dir + os.sep + 'clmf_kld_df' + '.csv'
if os.path.isfile(file_kld_df_csv):
    kld_df = pd.read_csv(file_kld_df_csv, dtype = {'snout_front': float, 'fl_l_front': float, 'fl_r_front': float, 
                                                   'snout_top': float, 'snout_bottom': float, 'fl_l': float, 'fl_r': float, 
                                                   'hl_l': float, 'hl_r': float, 'tailbase_bottom': float, 
                                                   'tailbase_top': float, 'day':str, 'mouse_id':str, 'group':str})
    
    kld_df = kld_df[kld_df['mouse_id'].isin( mice_list)]
    kld_df = kld_df[kld_df['group'].isin( groups)]
    days = kld_df.day.unique()
    for grp in np.unique(groups):
        print(grp)
        a = kld_df[kld_df['group'].isin([grp])].replace([np.nan, np.inf], 0)
        kld_days = np.stack([a[a.day==day][bp_select].mean() for day in days])
        fig, axs = plt.subplots(1,1, sharey=True, figsize=(5,3))
        im = sns.heatmap(kld_days.T, cmap='Greys', ax=axs)
        im.invert_yaxis()
        axs.set_xticklabels(days)
        axs.set_yticklabels(bp_select, rotation=0)
        plt.title(grp + ' Kullback-Leibler Divergence scores', fontsize=14)
        plt.tight_layout();
        pp.savefig(fig); plt.close()

# correlation matrix of paw speeds on each trial

file_df_speed_daily_all_pkl = out_dir + os.sep + 'clmf_df_speed_daily_all' + '.pkl'
file_df_beh_log_daily_all_pkl = out_dir + os.sep + 'clmf_df_beh_log_daily_all' + '.pkl'
df_speed_daily_all_file = open(file_df_speed_daily_all_pkl, 'rb')
df_speed_daily_all = pickle.load(df_speed_daily_all_file)
df_speed_daily_all_file.close()
df_beh_log_daily_all_file = open(file_df_beh_log_daily_all_pkl, 'rb')
df_beh_log_daily_all = pickle.load(df_beh_log_daily_all_file)
df_beh_log_daily_all_file.close()

# to select data for plotting
data_list = [('FW2_ai94', 'grp1'), 
             ('FW3_ai94', 'grp1'), 
             ('FW22_ai94', 'grp1'), 
             ('GT3_tta', 'grp1'), 
             ('GT33_tta', 'grp1'), ('HA2+_tta', 'grp1'), ('GER2_ai94', 'grp1'), ('HYL3_tta', 'grp1'), 
             ('GER2_ai94', 'grp2'), ('HYL3_tta', 'grp2'), 
             ('BR1_vgat-ai203', 'grp2'), ('BR2_vgat-ai203', 'grp2'), 
             ('GIL3_ai94', 'grp3'), 
             ('GIR2_ai94', 'grp3')]
groups = np.unique([d[1] for d in data_list])
days = [1,2,3,4,5,6,7,8,9,10]
beh_fps=30
bp_select = ['fl_l', 'fl_r', 'hl_l', 'hl_r', 'tailbase_bottom', 'tailbase_top']

## per-group: total bodypart movement during trial over days
for g in groups:
    bodypart_distance_grp_stack = []
    for expt in data_list:
        mouse_id, grp = expt
        if grp != g:
            continue
        bodyparts_distance_stack = []
        for day in days:
            df = df_beh_log_daily_all[(df_beh_log_daily_all['mouse_id']==mouse_id) & 
                                      (df_beh_log_daily_all['group']==grp) &
                                      (df_beh_log_daily_all['day']==day)]
            starts_ix = np.where(np.diff(df.trial.values) > 0)[0]
            ends_ix = np.where(np.diff(df.trial.values) < 0)[0]
            reward_ix = df[df.reward == 1].index
            fail_ix = df[df.reward == -1].index
            
            bodyparts_distance_stack.append(np.vstack([np.sum(df_speed_daily_all[bp_select].iloc[tr - 60:tr])/beh_fps for tr in reward_ix]).mean(axis=0))
        bodypart_distance_grp_stack.append(bodyparts_distance_stack)
    fig, axs = plt.subplots(1,1, sharey=True, figsize=(5,3))
    im = sns.heatmap(np.stack(bodypart_distance_grp_stack).mean(axis=0).T, cmap='Greys', ax=axs, vmin=10, vmax=40)
    im.invert_yaxis()
    axs.set_xticklabels(days)
    axs.set_yticklabels(bp_select, rotation=0)
    plt.title(g + ' bodyparts distances during trial')
    pp.savefig(fig); plt.close()
    
## speed correlation matrix for each group (average) during rewards
for g in groups:
    speed_corrmat_stack = []
    for expt in data_list:
        mouse_id, grp = expt
        if grp != g:
            continue
        fll_speed_stack = []
        flr_speed_stack = []
        for day in days:
            df = df_beh_log_daily_all[(df_beh_log_daily_all['mouse_id']==mouse_id) & 
                                      (df_beh_log_daily_all['group']==grp) &
                                      (df_beh_log_daily_all['day']==day)]
            starts_ix = np.where(np.diff(df.trial.values) > 0)[0]
            ends_ix = np.where(np.diff(df.trial.values) < 0)[0]
            reward_ix = df[df.reward == 1].index
            fail_ix = df[df.reward == -1].index
            fll_speed_stack.append(np.stack([df_speed_daily_all['fl_l'].iloc[tr - 60:tr] for tr in reward_ix]).mean(axis=0))
            flr_speed_stack.append(np.stack([df_speed_daily_all['fl_r'].iloc[tr - 60:tr] for tr in reward_ix]).mean(axis=0))
        speed_corrmat_stack.append(np.corrcoef(np.vstack((fll_speed_stack, flr_speed_stack))))
        
    fig, axs = plt.subplots(1,1, sharey=True)
    im = sns.heatmap(np.stack(speed_corrmat_stack).mean(axis=0), cmap='Greys', ax=axs, vmin=0.5, vmax=1)
    im.invert_yaxis()
    axs.set_xticklabels(np.append(['FLL_'+str(day) for day in days], ['FLR_'+str(day) for day in days]), rotation=90)
    axs.set_yticklabels(np.append(['FLL_'+str(day) for day in days], ['FLR_'+str(day) for day in days]), rotation=0)
    plt.title(g + '  speed profile correlations')
    pp.savefig(fig); plt.close()
## speed correlation matrix for each group (average) during rest
for g in groups:
    speed_corrmat_stack = []
    for expt in data_list:
        mouse_id, grp = expt
        if grp != g:
            continue
        fll_speed_stack = []
        flr_speed_stack = []
        for day in days:
            df = df_beh_log_daily_all[(df_beh_log_daily_all['mouse_id']==mouse_id) & 
                                      (df_beh_log_daily_all['group']==grp) &
                                      (df_beh_log_daily_all['day']==day)]
            starts_ix = np.where(np.diff(df.trial.values) > 0)[0]
            ends_ix = np.where(np.diff(df.trial.values) < 0)[0]
            reward_ix = df[df.reward == 1].index
            fail_ix = df[df.reward == -1].index
            fll_speed_stack.append(np.stack([df_speed_daily_all['fl_l'].iloc[tr - 60:tr] for tr in starts_ix]).mean(axis=0))
            flr_speed_stack.append(np.stack([df_speed_daily_all['fl_r'].iloc[tr - 60:tr] for tr in starts_ix]).mean(axis=0))
        speed_corrmat_stack.append(np.corrcoef(np.vstack((fll_speed_stack, flr_speed_stack))))

    fig, axs = plt.subplots(1,1, sharey=True)
    im = sns.heatmap(np.stack(speed_corrmat_stack).mean(axis=0), cmap='Greys', ax=axs, vmin=0, vmax=1, square=True)
    im.invert_yaxis()
    axs.set_xticklabels(np.append(['FLL_'+str(day) for day in days], ['FLR_'+str(day) for day in days]), rotation=90)
    axs.set_yticklabels(np.append(['FLL_'+str(day) for day in days], ['FLR_'+str(day) for day in days]), rotation=0)
    plt.title(g + '  speed profile correlations')
    plt.tight_layout()




#%%
data_list = [('FW2_ai94', 'grp1'), 
             ('FW22_ai94', 'grp1'), 
             ('GT33_tta', 'grp1'), ('HA2+_tta', 'grp1'), ('GER2_ai94', 'grp1'), ('HYL3_tta', 'grp1'), 
             ('GER2_ai94', 'grp2'), ('HYL3_tta', 'grp2'), 
             ('GIL3_ai94', 'grp3'), 
             ('GIR2_ai94', 'grp3')]
days = [1,4,5,7,10]
beh_fps = 30
epochSize = 5*beh_fps # should be same as used in generating the pkl file
t = np.arange(-epochSize, epochSize) / beh_fps

file_dff_response_all_pkl = out_dir + os.sep + 'clmf_avg_dff_response_daily' + '.pkl'
if not os.path.isfile(file_dff_response_all_pkl):
    sys.exit('DFF responses file doesnt exist')

dff_response_file = open(file_dff_response_all_pkl, 'rb')
dff_response_daily_all = pickle.load(dff_response_file)
dff_response_file.close()

#%% Plot average DFF activity for each seed location, per day for all mice
for expt in data_list:
    mouse_id, grp = expt
    for se in clmf_seeds_mm:
        print(mouse_id + ' ' + grp + ' ' + se)
        
        df = dff_response_daily_all[(dff_response_daily_all.mouse_id == mouse_id) &
                                    (dff_response_daily_all.group == grp) &
                                    (dff_response_daily_all.seedname == se)]
        if df.empty:
            print('No data')
            continue
        
        ## dff profiles at the seed location over the days
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12,5))
        axs[0].set_title(mouse_id + ' ' + grp + ' ' + se + ' L', fontsize=14)
        axs[1].set_title(mouse_id + ' ' + grp + ' ' + se + ' R', fontsize=14)
        g1=sns.lineplot(data=np.stack([df[df.day==day]['reward_stack_l'].values[0].mean(axis=0) for day in days]).T, ax=axs[0], legend=False, palette='Greens');
        # g1=sns.lineplot(data=np.stack([df[df.day==day]['rest_stack_l'].values[0].mean(axis=0) for day in days]).T, ax=axs[0], legend=False, palette='Reds');
        g2=sns.lineplot(data=np.stack([df[df.day==day]['reward_stack_r'].values[0].mean(axis=0) for day in days]).T, ax=axs[1], palette='Greens');
        # g2=sns.lineplot(data=np.stack([df[df.day==day]['rest_stack_r'].values[0].mean(axis=0) for day in days]).T, ax=axs[1], palette='Reds');
        plt.legend(days)
        plt.xlabel('Time (sec.)', fontweight='bold')
        # plt.xticks(np.arange(0, len(t),45), t[::45])
        g1.set_xticks(np.arange(0, len(t),60)) # <--- set the ticks first
        g1.set_xticklabels(t[::60])
        axs[0].axvline(epochSize, c='b')
        axs[0].axvline(epochSize-beh_fps, c='g')
        g2.set_xticks(np.arange(0, len(t),60)) # <--- set the ticks first
        g2.set_xticklabels(t[::60])
        axs[1].axvline(epochSize, c='b')
        axs[1].axvline(epochSize-beh_fps, c='g')
        plt.ylabel('DFF', fontweight='bold')
        sns.despine(offset=10, trim=True)
        # plt.yticks(fontsize=20)
        plt.tight_layout();
        pp.savefig(fig); plt.close()
        
        ## correlation matrix of dff responses at seedpixel over days
        fig, axs = plt.subplots(1,2, sharey=True)
        # im = axs[0].imshow(np.corrcoef(np.vstack([stk for stk in df['reward_stack_l']])), cmap='Greys');
        # im = axs[1].imshow(np.corrcoef(np.vstack([stk for stk in df['reward_stack_r']])), cmap='Greys');
        im = axs[0].imshow(np.corrcoef(np.vstack([stk.mean(axis=0) for stk in df['reward_stack_l']])), cmap='Greys');
        im = axs[1].imshow(np.corrcoef(np.vstack([stk.mean(axis=0) for stk in df['reward_stack_r']])), cmap='Greys');
        axs[0].set_title(mouse_id + ' ' + grp + ' ' + se + ' L', fontsize=14)
        axs[0].set_xticks(np.arange(len(df['day'].unique()))); axs[0].set_xticklabels(df['day'].unique())
        axs[0].set_yticks(np.arange(len(df['day'].unique()))); axs[0].set_yticklabels(df['day'].unique())
        axs[1].set_title(mouse_id + ' ' + grp + ' ' + se + ' R', fontsize=14)
        axs[1].set_xticks(np.arange(len(df['day'].unique()))); axs[1].set_xticklabels(df['day'].unique())
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
        pp.savefig(fig); plt.close()
        
        ## heatmap of dff responses at seedpixel over days
        minimum = np.nanmin([np.nanquantile(np.vstack([stk.mean(axis=0) for stk in df['reward_stack_l']]), 0.2), -0.001]); maximum = np.nanquantile(np.vstack([stk.mean(axis=0) for stk in df['reward_stack_l']]), 0.90)
        divnorm = colors.TwoSlopeNorm(vmin=minimum, vcenter=0, vmax=maximum)
        fig, axs = plt.subplots(1,2, sharey=True, figsize=(12,4))
        im = sns.heatmap(np.vstack([stk.mean(axis=0) for stk in df['reward_stack_l']]), cmap='coolwarm', vmin=minimum, vmax=maximum, center=0, ax=axs[0]);
        im = sns.heatmap(np.vstack([stk.mean(axis=0) for stk in df['reward_stack_r']]), cmap='coolwarm', vmin=minimum, vmax=maximum, center=0, ax=axs[1]);
        axs[0].set_title(mouse_id + ' ' + grp + ' ' + se + ' L', fontsize=14)
        axs[0].set_xlabel('Time (s)'); axs[0].set_ylabel('Day')
        axs[0].set_xticks(np.arange(0, len(t),30)); axs[0].set_xticklabels(t[::30])
        axs[0].set_yticklabels(df['day'].unique())
        axs[0].axvline(epochSize, c='k'); axs[0].axvline(epochSize-beh_fps, c='g')
        axs[1].set_title(mouse_id + ' ' + grp + ' ' + se + ' R', fontsize=14)
        axs[1].set_xlabel('Time (s)');
        axs[1].axvline(0, color='gray'); #axs[1].set_yticklabels(clmf_seeds_mm.keys())
        axs[1].set_xticks(np.arange(0, len(t),30)); axs[1].set_xticklabels(t[::30])
        axs[1].axvline(epochSize, c='k'); axs[1].axvline(epochSize-beh_fps, c='g')
        pp.savefig(fig); plt.close()
    
    ## seedpix corr during rewards on days
    for day in days:
        df = dff_response_daily_all[(dff_response_daily_all.mouse_id == mouse_id) &
                                    (dff_response_daily_all.group == grp) &
                                    (dff_response_daily_all.day == day)]
        
        fig, axs = plt.subplots(1,1, sharey=True)
        im = sns.heatmap(np.corrcoef(np.vstack([[stk.mean(axis=0) for stk in df['reward_stack_l']], [stk.mean(axis=0) for stk in df['reward_stack_r']]])), cmap='Greys', vmin=0, vmax=1);
        axs.set_title(mouse_id + ' ' + grp + ' ' + str(day) + '  seedpix corr rewards', fontsize=14)
        axs.set_xticklabels(np.concatenate([df.seedname.values + '_L', df.seedname.values + '_R']), rotation=90)
        axs.set_yticklabels(np.concatenate([df.seedname.values + '_L', df.seedname.values + '_R']), rotation=0)
        pp.savefig(fig); plt.close()
        
    ## heatmap of dff activity around reward on days
    df = dff_response_daily_all[(dff_response_daily_all.mouse_id == mouse_id) & (dff_response_daily_all.group == grp)]
    minimum = np.nanmin([np.nanquantile(np.vstack([stk.mean(axis=0) for stk in df['reward_stack_l']]), 0.2), -0.001]); maximum = np.nanquantile(np.vstack([stk.mean(axis=0) for stk in df['reward_stack_l']]), 0.90)
    divnorm = colors.TwoSlopeNorm(vmin=minimum, vcenter=0, vmax=maximum)
    for day in days:
        df = dff_response_daily_all[(dff_response_daily_all.mouse_id == mouse_id) &
                                    (dff_response_daily_all.group == grp) &
                                    (dff_response_daily_all.day == day)]
        
        fig, axs = plt.subplots(1,1, sharey=True)
        im = sns.heatmap(np.vstack([[stk.mean(axis=0) for stk in df['reward_stack_l']], [stk.mean(axis=0) for stk in df['reward_stack_r']]]), cmap='Greys', vmin=minimum, vmax=maximum);
        axs.set_title(mouse_id + ' ' + grp + ' ' + str(day) + '  dff response', fontsize=14)
        axs.set_xticks(np.arange(0, len(t),30)); axs.set_xticklabels(t[::30])
        axs.set_yticklabels(np.concatenate([df.seedname.values + '_L', df.seedname.values + '_R']), rotation=0)
        axs.axvline(epochSize, c='c'); axs.axvline(epochSize-beh_fps, c='g')
        pp.savefig(fig); plt.close()
        
#%% Plot max dff values before reward and before trial start for all mice combined
data_list = [('FW2_ai94', 'grp1'), 
             ('FW22_ai94', 'grp1'), 
             ('GT33_tta', 'grp1'), ('HA2+_tta', 'grp1'), ('GER2_ai94', 'grp1'), ('HYL3_tta', 'grp1'), 
             ('GER2_ai94', 'grp2'), ('HYL3_tta', 'grp2'), 
             ('GIR2_ai94', 'grp3')]
days = [1,2,3,4,5,6,7,8,9,10]
# First build a dataframe with max dff values for all mice in all groups we can plot by groups later
for expt in data_list:
    mouse_id, grp = expt
    for se in clmf_seeds_mm:
        for day in days:
            df = dff_response_daily_all.loc[(dff_response_daily_all.mouse_id == mouse_id) &
                                       (dff_response_daily_all.group == grp) &
                                       (dff_response_daily_all.day == day) &
                                       (dff_response_daily_all.seedname == se)]
            if df.empty:
                print('No data')
                continue
            
            
            dff_response_daily_all.loc[(dff_response_daily_all.mouse_id == mouse_id) &
                                        (dff_response_daily_all.group == grp) &
                                        (dff_response_daily_all.day == day) &
                                        (dff_response_daily_all.seedname == se), ['max_prereward_avg_l']] = np.max(df['reward_stack_l'].values[0].mean(axis=0)[int(epochSize/2):epochSize])
            dff_response_daily_all.loc[(dff_response_daily_all.mouse_id == mouse_id) &
                                        (dff_response_daily_all.group == grp) &
                                        (dff_response_daily_all.day == day) &
                                        (dff_response_daily_all.seedname == se), ['max_prereward_avg_r']] = np.max(df['reward_stack_r'].values[0].mean(axis=0)[int(epochSize/2):epochSize])
            dff_response_daily_all.loc[(dff_response_daily_all.mouse_id == mouse_id) &
                                        (dff_response_daily_all.group == grp) &
                                        (dff_response_daily_all.day == day) &
                                        (dff_response_daily_all.seedname == se), ['max_postreward_avg_l']] = np.max(df['reward_stack_l'].values[0].mean(axis=0)[epochSize:epochSize+int(epochSize/2)])
            dff_response_daily_all.loc[(dff_response_daily_all.mouse_id == mouse_id) &
                                        (dff_response_daily_all.group == grp) &
                                        (dff_response_daily_all.day == day) &
                                        (dff_response_daily_all.seedname == se), ['max_postreward_avg_r']] = np.max(df['reward_stack_r'].values[0].mean(axis=0)[epochSize:epochSize+int(epochSize/2)])
            dff_response_daily_all.loc[(dff_response_daily_all.mouse_id == mouse_id) &
                                        (dff_response_daily_all.group == grp) &
                                        (dff_response_daily_all.day == day) &
                                        (dff_response_daily_all.seedname == se), ['max_rest_avg_l']] = np.max(df['rest_stack_l'].values[0].mean(axis=0)[int(epochSize/2):epochSize])
            dff_response_daily_all.loc[(dff_response_daily_all.mouse_id == mouse_id) &
                                        (dff_response_daily_all.group == grp) &
                                        (dff_response_daily_all.day == day) &
                                        (dff_response_daily_all.seedname == se), ['max_rest_avg_r']] = np.max(df['rest_stack_r'].values[0].mean(axis=0)[int(epochSize/2):epochSize])
            
mouse_list = [d[0] for d in data_list]
grp = 'grp1'
for se in clmf_seeds_mm:
    df = dff_response_daily_all[(dff_response_daily_all.mouse_id.isin(mouse_list)) &
                                (dff_response_daily_all.day.isin(days)) &
                                (dff_response_daily_all.group == grp) & 
                                (dff_response_daily_all.seedname == se)]
    dfl = df.melt(id_vars=['day'], value_vars=['max_prereward_avg_l', 'max_rest_avg_l'], var_name='condition', value_name='DFF')
    dfr = df.melt(id_vars=['day'], value_vars=['max_prereward_avg_r', 'max_rest_avg_r'], var_name='condition', value_name='DFF')

    hue_plot_params = {
        'data': dfl,
        'x': 'day',
        'y': 'DFF',
        "hue": "condition",
        "palette": 'husl'
    }
    pairs = [
    [(1, 'max_prereward_avg_l'), (4, 'max_prereward_avg_l')],
    [(4, 'max_prereward_avg_l'), (5, 'max_prereward_avg_l')]
    ]
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12,6))
    ax1 = sns.pointplot(**hue_plot_params, dodge=0.2, errorbar=('ci', 95), ax=axs[0])
    ax2 = sns.pointplot(x='day', y='DFF', data=dfr, hue='condition', palette='husl', dodge=0.2, errorbar=('ci', 95), ax=axs[1])
    plt.title(grp + ' ' + se)
    sns.despine(offset=10, trim=True)
    # plt.yticks(fontsize=20)
    plt.tight_layout();
    # Annotate the figure with stats
    annotator = Annotator(ax1, pairs, **hue_plot_params)
    # # test: Brunner-Munzel, Levene, Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, t-test_ind, t-test_welch, t-test_paired, Wilcoxon, Kruskal
    # # comparisons_correction: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg, Benjamini-Yekutieli
    annotator.configure(test='Mann-Whitney', show_test_name=True, comparisons_correction='Benjamini-Hochberg', text_format='star') #bejamini hochberg, Bonferroni
    _, test_results = annotator.apply_and_annotate()
    for res in test_results: print(res.data)
    pp.savefig(fig); plt.close()







file_corrmat_daily_all_pkl = out_dir + os.sep + 'clmf_corrmat_daily_all' + '.pkl'
if not os.path.isfile(file_corrmat_daily_all_pkl):
    sys.exit('Correlation matrices file doesnt exist')
corrmat_file = open(file_corrmat_daily_all_pkl, 'rb')
df_corrmat_daily_all = pickle.load(corrmat_file)
corrmat_file.close()
 
bregma_loc = {'x': 64, 'y': 64}
bregma = Position(bregma_loc['y'], bregma_loc['x'])
seeds = generate_seeds(bregma, clmf_seeds_mm, ppmm_brain, 'u')
    
#%% p-value matrix of statistical test
data_list = [
            ('FW2_ai94', 'grp1'), ('FW3_ai94', 'grp1'), ('FW22_ai94', 'grp1'), 
             ('GT33_tta', 'grp1'), ('HA2+_tta', 'grp1'), ('GER2_ai94', 'grp1'),
              ('GIL3_ai94', 'grp3'), ('GIR2_ai94', 'grp3')
             ]
days = [1,2,3,4,5,6,7,8,9,10]
mouse_list = [d[0] for d in data_list]
groups = np.unique([d[1] for d in data_list])

# Supplementary figure 3 average corrmat during trial on all days, each group
for grp in groups:

    aa1 = np.mean(df_corrmat_daily_all[(df_corrmat_daily_all.mouse_id.isin(mouse_list)) & 
                                        (df_corrmat_daily_all.group == grp)]['reward_corrmat'].values)
    aa2 = np.mean(df_corrmat_daily_all[(df_corrmat_daily_all.mouse_id.isin(mouse_list)) & 
                                        (df_corrmat_daily_all.group == grp)]['rest_corrmat'].values)
    
    aa3 = aa1-aa2
    vmin = np.nanquantile(aa1, 0.2); vmax = np.quantile(aa3, 0.95)
    ## plot each of the difference matrix in the stack we created above
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1); ax=sns.heatmap(aa1, center=0, vmin=vmin, vmax=vmax, linewidths=.5, 
                                         cmap='coolwarm', square=True, xticklabels=seeds.keys(), 
                                         yticklabels=seeds.keys(), cbar_kws={"shrink":.5, "orientation":'horizontal'})
    plt.title(grp + ' mean seedpixel correlation during trial, all days')
    plt.tight_layout(); pp.savefig(fig); plt.close()
    
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1); ax=sns.heatmap(aa2, center=0, vmin=vmin, vmax=vmax, linewidths=.5, 
                                         cmap='coolwarm', square=True, xticklabels=seeds.keys(), 
                                         yticklabels=seeds.keys(), cbar_kws={"shrink":.5, "orientation":'horizontal'})
    plt.title(grp + ' mean seedpixel correlation during rest, all days')
    plt.tight_layout(); pp.savefig(fig); plt.close()
    
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1); ax=sns.heatmap(aa3, center=0, vmin=vmin, vmax=vmax, linewidths=.5, 
                                         cmap='coolwarm', square=True, xticklabels=seeds.keys(), 
                                         yticklabels=seeds.keys(), cbar_kws={"shrink":.5, "orientation":'horizontal'})
    plt.title(grp + ' difference of correlations between trial and rest, all days')
    plt.tight_layout(); pp.savefig(fig); plt.close()
    
#%% Supplementary figure 3
df_seeds_corrmat = pd.DataFrame()
for expt in data_list:
    mouse_id, grp = expt

    for day in days:
        df = df_corrmat_daily_all[(df_corrmat_daily_all.mouse_id == mouse_id) & 
                                  (df_corrmat_daily_all.group == grp) &
                                  (df_corrmat_daily_all.day == day)]
        if df.empty:
            print(mouse_id + ' ' + grp + ' No data')
            continue
        reward_corrmat = df['reward_corrmat'].values[0]
        rest_corrmat = df['rest_corrmat'].values[0]
        
        for i, se1 in enumerate(seeds):
            for j, se2 in enumerate(seeds):
                df_seeds_corrmat = pd.concat([df_seeds_corrmat, pd.DataFrame({'mouse_id': mouse_id, 'group': grp, 
                                                                 'day': day, 'condition': 'trial', 
                                                                 'seed1': se1, 'seed2': se2, 
                                                                 'correlation': reward_corrmat[i,j]}, index=[0])], 
                                             ignore_index=True)
                df_seeds_corrmat = pd.concat([df_seeds_corrmat, pd.DataFrame({'mouse_id': mouse_id, 'group': grp, 
                                                                 'day': day, 'condition': 'rest', 
                                                                 'seed1': se1, 'seed2': se2, 
                                                                 'correlation': rest_corrmat[i,j]}, index=[0])], 
                                             ignore_index=True)

for grp in groups:
    # initialize dict to hold pvalues of statistical test anova
    pvalue_day_corrmat = np.ones((len(seeds), len(seeds)))
    pvalue_condition_corrmat = np.ones((len(seeds), len(seeds)))
    for i, se1 in enumerate(seeds):
        for j, se2 in enumerate(seeds):
            df = df_seeds_corrmat[(df_seeds_corrmat['seed1']==se1) 
                                  & (df_seeds_corrmat['seed2']==se2) 
                                  & (df_seeds_corrmat['group']==grp) 
                                  # & (df_seeds_corrmat['condition']=='rest')
                                  ].reset_index()
            # repeated measure ANOVA because we have samples over days. this is two way anova 
            # becasue we are adding condition as a variable too
            res = pg.rm_anova(dv='correlation', within=['day', 'condition'], subject='mouse_id', data=df, detailed=True)
            if 'p-unc' in res:
                pvalue_day_corrmat[i,j] = res['p-unc'][0]
                pvalue_condition_corrmat[i,j] = res['p-unc'][1]
    
    #### bonferroni correction
    pvalue_day_corrmat_corrected = multipletests(pvalue_day_corrmat.flatten(), method='bonferroni')
    fig = plt.figure(); ax=sns.heatmap(pvalue_day_corrmat_corrected[1].reshape(pvalue_day_corrmat.shape), linewidths=.5, cmap='Greens_r', square=True, 
                                       vmin= 0.0001, vmax = 0.06, xticklabels=seeds.keys(), yticklabels=seeds.keys(), 
                                       cbar_kws={"shrink":.5, "orientation":'horizontal'})
    plt.title(grp + ' RM-ANOVA pvalues on correlation values over days')
    pp.savefig(fig); plt.close()
    
    #### bonferroni correction
    pvalue_condition_corrmat_corrected = multipletests(pvalue_condition_corrmat.flatten(), method='bonferroni')
    fig = plt.figure(); ax=sns.heatmap(pvalue_condition_corrmat_corrected[1].reshape(pvalue_condition_corrmat.shape), linewidths=.5, cmap='Greens_r', square=True, 
                                       vmin= 0.0001, vmax = 0.06, xticklabels=seeds.keys(), yticklabels=seeds.keys(), 
                                       cbar_kws={"shrink":.5, "orientation":'horizontal'})
    plt.title(grp + ' RM-ANOVA pvalues on correlation values between trial and rest, over days')
    pp.savefig(fig); plt.close()


print('done')
pp.close()