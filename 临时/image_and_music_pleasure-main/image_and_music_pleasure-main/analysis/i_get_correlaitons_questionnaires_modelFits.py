# -*- coding: utf-8 -*-
"""
Created on Wed Dec 8, 2021

@author: abrielmann

 look at potential relationships between AReA and BMQR and
the model fits/parameters
"""
import os
import pandas as pd
import scipy.stats

# 设置目录
homeDir = 'D:\\临时\\image_and_music_pleasure-main\\image_and_music_pleasure-main\\data'
experiment = 'exp4'
dataDir = os.path.join(homeDir, experiment)

#%% fetch dattta
df = pd.read_csv(dataDir + '/results_per_participant.csv')

firstResCol = df.columns.get_loc('avgRmse_accurate_preOne')

firstScoreCol = df.columns.get_loc('AREA_1')

# 创建包含所有评分列的 DataFrame
scoreDf = df.iloc[:, firstScoreCol:firstScoreCol+11]

# 创建包含所有结果列的 DataFrame
resDf = df.iloc[:, firstResCol:]

scoreNames = scoreDf.columns
resNames = resDf.columns


#%% we loop through all scores and all results and correlate
scoreList = []
resList = []
rhoList = []
pList = []
for score in scoreNames:
    for res in resNames:
        rho, pval = scipy.stats.spearmanr(df[score], df[res])
        scoreList.append(score)
        resList.append(res)
        rhoList.append(rho)
        pList.append(pval)

# merge those results into a df for easier handling, saving
corrDict = {'score': scoreList, 'res': resList, 'rho': rhoList, 'p': pList}
corrDf = pd.DataFrame(corrDict)

#%% display correlations that are, by standard cut-offs, significant
print(corrDf[corrDf.p<.05])

#%% the above is a long table for exp3 --> pack it into a df, save as csv
sigCorr = corrDf[corrDf.p<.05]
sigCorr.to_csv(dataDir + '/significantCorrelations_questionnaires_fits.csv')

