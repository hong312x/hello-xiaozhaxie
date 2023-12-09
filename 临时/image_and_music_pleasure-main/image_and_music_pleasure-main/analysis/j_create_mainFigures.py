# -*- coding: utf-8 -*-
"""
Created on Mon Dec 6, 2021

@author: abrielmann

"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# import and update global figure settings
import matplotlib.pylab as pylab
params = {'legend.fontsize': 10,
          'legend.title_fontsize': 12,
          'legend.borderpad': 0,
          'figure.figsize': (8,10),
         'axes.labelsize': 10,
         'axes.titlesize': 12,
         'xtick.labelsize': 10,
         'ytick.labelsize': 10,
         'lines.linewidth': 2,
         'image.cmap': 'gray',
         'savefig.dpi': 300}
pylab.rcParams.update(params)

# 设置目录
homeDir = 'D:\\临时\\image_and_music_pleasure-main\\image_and_music_pleasure-main\\data'
experiment = 'exp4'
dataDir = os.path.join(homeDir, experiment)

#%% fetch pre-processed data
rawDf = pd.read_csv(dataDir + '/all_data.csv')
resDfPrecue = pd.read_csv(dataDir + '/res_one_Precue.csv')
resDfPrecue['cue'] = 'Pre'
resDfPrecue['task'] = 'rateOne'
resDfPostcue = pd.read_csv(dataDir + '/res_one_Postcue.csv')
resDfPostcue['cue'] = 'Post'
resDfPostcue['task'] = 'rateOne'

resDfPrecueBoth = pd.read_csv(dataDir + '/res_both_Precue.csv')
resDfPrecueBoth['cue'] = 'Pre'
resDfPrecueBoth['task'] = 'rateBoth'
resDfPostcueBoth = pd.read_csv(dataDir + '/res_both_Postcue.csv')
resDfPostcueBoth['cue'] = 'Post'
resDfPostcueBoth['task'] = 'rateBoth'

# merge result DFs
resDf = pd.concat([resDfPrecue, resDfPostcue, resDfPrecueBoth, resDfPostcueBoth])
rmseDf = pd.wide_to_long(resDf.reset_index(), ['avgRMSE'],
                         i=['participant','cue', 'task'],
                         j='model', sep='_', suffix=r'\w+')
rmseDf.reset_index(inplace=True)

# get cronbach's; corrs
relDf = pd.read_csv(dataDir + '/reliabilities_correlations.csv')

# manual entry of reported alphas and correlations in Brielmann & Pelli (2020)
conditions = ['one precued', 'one postcued', 'both precued', 'both postcued']
alpha_2images = [0.92, 0.89, 0.85, 0.81]
relations = ['one pre vs post', 'both pre vs post', 'pre one vs both',
             'post one vs both']
corrs_2images = [0.93, 0.92, 0.52, 0.45]

