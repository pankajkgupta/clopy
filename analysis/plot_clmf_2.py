#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:09:03 2024
Requires file clmf_df_speed_daily_all.pkl clmf_df_beh_log_daily_all.pkl, clmf_avg_dff_response_daily.pkl
Please download from here-https://drive.google.com/drive/folders/1eq0muLtc36jcU4sUy8fVOSmN4dxVxdn8?usp=sharing
@author: pankaj gupta
"""
import os
import platform
import sys
from os.path import dirname, realpath
import cv2

import matplotlib.pyplot as plt
import numpy as np

filepath = realpath(__file__)
dir_of_file = dirname(filepath)
parent_dir_of_file = dirname(dir_of_file)
sys.path.append(parent_dir_of_file)
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from matplotlib import colors
import matplotlib
from matplotlib import colors
matplotlib.use('agg')
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import ImageGrid
import tifffile as tif
import pandas as pd
import pickle
import seaborn as sns
import seaborn_image as snsi
from pathlib import Path
import glob
from tqdm import tqdm
from scipy import signal
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from skimage.util import montage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import accuracy_score
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.deep_learning.cnn import CNNClassifier
from statannotations.Annotator import Annotator
import imageio
import argparse
from configparser import ConfigParser
from sty import fg, bg, ef, rs
os.environ[ 'NUMBA_CACHE_DIR' ] = '/tmp/numba_cache/'
import dlc2kinematics
# import napari
from dtaidistance import dtw
from get_clmf_data import \
    expt1_data_list, expt2_data_list, expt3_data_list
from helper import *

data_root = '/home/pankaj/teamshare/pkg/closedloop_rig5_data/'
out_dir = '../data/'

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

pp = PdfPages(out_dir + os.path.basename(__file__).split('.')[0] + '_stats.pdf')
#%% Plot average DFF activity for each seed location, per day for all mice
for expt in data_list:
    mouse_id, grp = expt
    for se in seeds_mm:
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
        axs[1].axvline(0, color='gray'); #axs[1].set_yticklabels(seeds_mm.keys())
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
    for se in seeds_mm:
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
for se in seeds_mm:
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
    ax1 = sns.pointplot(**hue_plot_params, dodge=0.2, errorbar=('ci', 95), ax=axs[0], linewidth=1, markersize=5)
    ax2 = sns.pointplot(x='day', y='DFF', data=dfr, hue='condition', palette='husl', dodge=0.2, errorbar=('ci', 95), ax=axs[1], linewidth=1, markersize=5)
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


print('done')
pp.close()