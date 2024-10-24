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
import pingouin as pg

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
from sklearn.metrics.pairwise import euclidean_distances
import imageio
import argparse
from configparser import ConfigParser
from sty import fg, bg, ef, rs
os.environ[ 'NUMBA_CACHE_DIR' ] = '/tmp/numba_cache/'
import dlc2kinematics
from statsmodels.stats.multitest import multipletests
# import napari
from dtaidistance import dtw
from helper import *

data_root = '/home/pankaj/teamshare/pkg/closedloop_rig5_data/'
out_dir = '../processed_data/'

# to select data for plotting
data_list = [
    ('FW2_ai94', 'grp1'), ('FW3_ai94', 'grp1'), ('FW22_ai94', 'grp1'), ('GT3_tta', 'grp1'), 
    ('GT33_tta', 'grp1'), ('HA2+_tta', 'grp1'), ('GER2_ai94', 'grp1'), ('HYL3_tta', 'grp1'), 
    ('GER2_ai94', 'grp2'), ('HYL3_tta', 'grp2'), 
    ('GIL3_ai94', 'grp3'), ('GIR2_ai94', 'grp3')
             ]
days = [1,4,5,7,10]

file_corrmat_daily_all_pkl = out_dir + os.sep + 'clmf_corrmat_daily_all' + '.pkl'
if not os.path.isfile(file_corrmat_daily_all_pkl):
    sys.exit('Correlation matrices file doesnt exist')
    
corrmat_file = open(file_corrmat_daily_all_pkl, 'rb')
df_corrmat_daily_all = pickle.load(corrmat_file)
corrmat_file.close()

ppmm_brain = 14.56
pp = PdfPages(out_dir + os.path.basename(__file__).split('.')[0] + '_summary_stats.pdf')
for expt in data_list:
    mouse_id, grp = expt
    
    bregma_loc = {'x': 64, 'y': 64}
    bregma = Position(bregma_loc['y'], bregma_loc['x'])
    seeds = generate_seeds(bregma, clmf_seeds_mm, ppmm_brain, 'u')
    ### this block plots corrmat during trial, rest and their difference on each day
    for day in days:
        print(mouse_id + ' ' + grp + ' ' + str(day))
        df = df_corrmat_daily_all[(df_corrmat_daily_all.mouse_id == mouse_id) &
                                  (df_corrmat_daily_all.group == grp) &
                                  (df_corrmat_daily_all.day == day)]
        if df.empty:
            continue
        fig = plt.figure(figsize=(15, 10))
        plt.subplot(1, 3, 1); sns.heatmap(df.reward_corrmat.values[0], center=0, linewidths=.5, cmap='coolwarm', 
                                          vmin=0, vmax=1, square=True, xticklabels=seeds.keys(),
                                          yticklabels=seeds.keys(), cbar_kws={"shrink": .5, "orientation":'horizontal'})
        plt.title(mouse_id + ' ' + grp + ' ' + str(day )+ ' reward corr.')
        plt.subplot(1, 3, 2); sns.heatmap(df.rest_corrmat.values[0], center=0, linewidths=.5, cmap='coolwarm', 
                                          vmin=0, vmax=1, square=True, xticklabels=seeds.keys(),
                                          yticklabels=seeds.keys(), cbar_kws={"shrink": .5, "orientation":'horizontal'})
        plt.title(mouse_id + ' ' + grp + ' ' + str(day)+ ' rest corr.')
        plt.subplot(1, 3, 3); sns.heatmap(df.reward_corrmat.values[0]-df.rest_corrmat.values[0], center=0, linewidths=.5, cmap='coolwarm', 
                                          square=True, xticklabels=seeds.keys(),
                                          yticklabels=seeds.keys(), cbar_kws={"shrink": .5, "orientation":'horizontal'})
        plt.title(mouse_id + ' ' + grp + ' ' + str(day) + ' reward - rest corr.')
        plt.tight_layout(); pp.savefig(fig); plt.close()
        
    
    # %% plot difference of corrmat during trial on each day
    df = df_corrmat_daily_all[(df_corrmat_daily_all.mouse_id == mouse_id) & 
                              (df_corrmat_daily_all.group == grp)]
    if df.empty:
        print(mouse_id + ' ' + grp + ' No data')
        continue
    
    diffrewardmat = np.stack(df['reward_corrmat'].values); diffrewardmat = np.diff(diffrewardmat, axis=0)
    vmin = np.nanquantile(diffrewardmat, 0.2); vmax = np.quantile(diffrewardmat, 0.95)
    fig = plt.figure(figsize=(18, 6))
    for i in np.arange(diffrewardmat.shape[0]):
        plt.subplot(1, diffrewardmat.shape[0], i+1); ax=sns.heatmap(diffrewardmat[i], center=0, vmin=vmin, vmax=vmax, linewidths=.5, cmap='coolwarm', square=True,
                                                      xticklabels=seeds.keys(), yticklabels=seeds.keys(), cbar_kws={"shrink":.5, "orientation":'horizontal'})
    plt.title(mouse_id + ' ' + grp + ' difference of correlations during trial, over days')
    plt.tight_layout(); pp.savefig(fig); plt.close()
    # %% plot difference of corrmat during rest on each day
    df = df_corrmat_daily_all[(df_corrmat_daily_all.mouse_id == mouse_id) & 
                              (df_corrmat_daily_all.group == grp)]
    if df.empty:
        print(mouse_id + ' ' + grp + ' No data')
        continue
    
    diffrestmat = np.stack(df['rest_corrmat'].values); diffrestmat = np.diff(diffrestmat, axis=0)
    vmin = np.nanquantile(diffrestmat, 0.2); vmax = np.quantile(diffrestmat, 0.95)
    fig = plt.figure(figsize=(18, 6))
    for i in np.arange(diffrestmat.shape[0]):
        plt.subplot(1, diffrestmat.shape[0], i+1); ax=sns.heatmap(diffrestmat[i], center=0, vmin=vmin, vmax=vmax, linewidths=.5, cmap='coolwarm', square=True,
                                                      xticklabels=seeds.keys(), yticklabels=seeds.keys(), cbar_kws={"shrink":.5, "orientation":'horizontal'})
    plt.title(mouse_id + ' ' + grp + ' difference of correlations during rest, over days')
    plt.tight_layout(); pp.savefig(fig); plt.close()
    #%% vector operation: subtract correlation matrices during trial and rest on all days, stack the result
    diffmat = np.stack(df.reward_corrmat.values) - np.stack(df.rest_corrmat.values)
    vmin = np.nanquantile(diffmat, 0.2); vmax = np.quantile(diffmat, 0.95)
    ## plot each of the difference matrix in the stack we created above
    fig = plt.figure(figsize=(18, 6))
    for i in np.arange(diffmat.shape[0]):
        plt.subplot(1, diffmat.shape[0], i+1); ax=sns.heatmap(diffmat[i], center=0, vmin=vmin, vmax=vmax, linewidths=.5, cmap='coolwarm', square=True,
                                                      xticklabels=seeds.keys(), yticklabels=seeds.keys(), cbar_kws={"shrink":.5, "orientation":'horizontal'})
    plt.title(mouse_id + ' ' + grp + ' difference of correlations between trial and rest, over days')
    plt.tight_layout(); pp.savefig(fig); plt.close()
    
#%% p-value matrix of statistical test
data_list = [
            ('FW2_ai94', 'grp1'), ('FW3_ai94', 'grp1'), ('FW22_ai94', 'grp1'), 
             ('GT33_tta', 'grp1'), ('HA2+_tta', 'grp1'), ('GER2_ai94', 'grp1'),
              ('GIL3_ai94', 'grp3'), ('GIR2_ai94', 'grp3')
             ]
days = [1,2,3,4,5,6,7,8,9,10]
mouse_list = [d[0] for d in data_list]
groups = np.unique([d[1] for d in data_list])

# difference of average corrmats during task and rest on each day
for grp in groups:

    aa1 = np.stack([np.mean(df_corrmat_daily_all[(df_corrmat_daily_all.mouse_id.isin(mouse_list)) & 
                                        (df_corrmat_daily_all.group == grp) &
                                        (df_corrmat_daily_all.day == day)]['reward_corrmat'].values) for day in days])
    aa2 = np.stack([np.mean(df_corrmat_daily_all[(df_corrmat_daily_all.mouse_id.isin(mouse_list)) & 
                                        (df_corrmat_daily_all.group == grp) &
                                        (df_corrmat_daily_all.day == day)]['rest_corrmat'].values) for day in days])
    aa3 = aa1-aa2
    vmin = np.nanquantile(aa3, 0.2); vmax = np.quantile(aa3, 0.95)
    ## plot each of the difference matrix in the stack we created above
    fig = plt.figure(figsize=(18, 6))
    for i in np.arange(aa3.shape[0]):
        plt.subplot(1, aa3.shape[0], i+1); ax=sns.heatmap(aa3[i], center=0, vmin=vmin, vmax=vmax, linewidths=.5, cmap='coolwarm', square=True,
                                                      xticklabels=seeds.keys(), yticklabels=seeds.keys(), cbar_kws={"shrink":.5, "orientation":'horizontal'})
    plt.title(grp + ' difference of correlations between trial and rest, over days')
    plt.tight_layout(); pp.savefig(fig); plt.close()

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

pp.close()