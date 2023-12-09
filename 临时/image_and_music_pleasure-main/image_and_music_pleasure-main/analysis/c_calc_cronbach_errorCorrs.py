# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:26:16 2021

@author: abrielmann
"""
import os
import numpy as np
import pandas as pd
import pingouin as pg

# 设置目录
homeDir = 'D:\\临时\\image_and_music_pleasure-main\\image_and_music_pleasure-main\\data'
experiment = 'exp4'
dataDir = os.path.join(homeDir, experiment)

#%% fetch pre-processed data
df = pd.read_csv(dataDir + '/all_data.csv')
participants = np.unique(df.participant)
nParticipants = len(participants)

# 确保 df.image 列中的所有值都是字符串
df['image'] = df['image'].astype(str)
# 然后尝试使用 np.unique
images = np.unique(df.image)

# 将 df.music 列中的所有值转换为字符串
df['music'] = df['music'].astype(str)
# 然后尝试使用 np.unique
music = np.unique(df.music)

stims = np.concatenate((images, music))
nImages = len(images)
nSongs = len(music)
nStims = nImages+nSongs

#%% Loop through participants and get 'errors', i.e., deviation of rating
# compared to baseline per image per cueing condition

err_pre_one = np.empty((nStims, nParticipants))
err_post_one = np.empty((nStims, nParticipants))
err_pre_both = np.empty((nStims, nParticipants))
err_post_both = np.empty((nStims, nParticipants))

countPeeps = 0
for peep in participants:
    peepsDf = df.loc[df['participant']==peep]

    countStims = 0
    for stim in stims:
        stimInd = (peepsDf['cuedImage']==stim) | (peepsDf['cuedMusic']==stim)
        thisDf = peepsDf.loc[stimInd]
        precueDf = thisDf.loc[thisDf['cueTime']=='Pre']
        postcueDf = thisDf.loc[thisDf['cueTime']=='Post']

        average_value = precueDf.loc[precueDf['cued'] != 'Both', 'diffRatePred'].mean()
        err_pre_one[countStims, countPeeps] = average_value

        average_value = postcueDf.loc[postcueDf['cued'] != 'Both', 'diffRatePred'].mean()
        err_post_one[countStims, countPeeps] = average_value

        err_pre_both[countStims, countPeeps] = np.mean(np.abs(precueDf.loc[precueDf['cued']=='Both', 'diffRatePred']))
        err_post_both[countStims, countPeeps] = np.mean(np.abs(postcueDf.loc[postcueDf['cued']=='Both', 'diffRatePred']))

        countStims += 1

    countPeeps += 1

# %% With these 'errors' calculated, we can now compute cronbach's alpha for
# each condition
alpha_pre_one, alpha_pre_one_CI = pg.cronbach_alpha(pd.DataFrame(err_pre_one.T))
alpha_post_one, alpha_post_one_CI = pg.cronbach_alpha(pd.DataFrame(err_post_one.T))

alpha_pre_both, alpha_pre_both_CI = pg.cronbach_alpha(pd.DataFrame(err_pre_both.T))
alpha_post_both, alpha_post_both_CI = pg.cronbach_alpha(pd.DataFrame(err_post_both.T))

alphas = [alpha_pre_one, alpha_post_one, alpha_pre_both, alpha_post_both]
alpha_negErr = [alpha_pre_one-alpha_pre_one_CI[0],
              alpha_post_one-alpha_post_one_CI[0],
              alpha_pre_both-alpha_pre_both_CI[0],
              alpha_post_both-alpha_post_both_CI[0]]
alpha_posErr = [alpha_pre_one_CI[1]-alpha_pre_one,
              alpha_post_one_CI[1]-alpha_post_one,
              alpha_pre_both_CI[1]-alpha_pre_both,
              alpha_post_both_CI[1]-alpha_post_both]
alpha_lowerCIs = [alpha_pre_one_CI[0],
              alpha_post_one_CI[0],
              alpha_pre_both_CI[0],
              alpha_post_both_CI[0]]
alpha_upperCIs = [alpha_pre_one_CI[1],
              alpha_post_one_CI[1],
              alpha_pre_both_CI[1],
              alpha_post_both_CI[1]]

#%% the alphas above also provide an upper bound to possible correlations
# between errors across trial types
max_r_prePost_one = np.sqrt(alpha_pre_one*alpha_post_one)
max_r_prePost_both = np.sqrt(alpha_pre_both*alpha_post_both)
max_r_oneBoth_pre = np.sqrt(alpha_pre_one*alpha_pre_both)
max_r_oneBoth_post = np.sqrt(alpha_post_one*alpha_post_both)


def safe_corr(x, y):
    if len(x) >= 2 and len(y) >= 2 and not np.all(x == x[0]) and not np.all(y == y[0]):
        return pg.corr(x, y)
    else:
        return None


def check_and_compute_corr(arr1, arr2):
    if len(arr1) >= 2 and len(arr2) >= 2:
        return safe_corr(arr1, arr2)
    else:
        return None
# 删除 NaN 值的示例
flattened_pre_one = np.nan_to_num(err_pre_one).flatten()
flattened_post_one = np.nan_to_num(err_post_one).flatten()
flattened_pre_both = np.nan_to_num(err_pre_both).flatten()
flattened_post_both = np.nan_to_num(err_post_both).flatten()

# 使用安全的相关性计算函数
corr_prePost_one = check_and_compute_corr(flattened_pre_one, flattened_post_one)
corr_prePost_both = check_and_compute_corr(flattened_pre_both, flattened_post_both)
corr_oneBoth_pre = check_and_compute_corr(flattened_pre_one, flattened_pre_both)
corr_oneBoth_post = check_and_compute_corr(flattened_post_one, flattened_post_both)

# 在提取相关性结果和置信区间之前，进行检查
corr_negErr = []
corr_posErr = []
for corr in [corr_prePost_one, corr_prePost_both, corr_oneBoth_pre, corr_oneBoth_post]:
    if corr is not None:
        try:
            neg_err = corr['r'].values[0] - corr['CI95%'][0][0]
            pos_err = corr['CI95%'][0][1] - corr['r'].values[0]
            corr_negErr.append(neg_err)
            corr_posErr.append(pos_err)
        except IndexError:
            # 处理无效的索引
            print("Invalid index when accessing correlation results.")
            corr_negErr.append(None)
            corr_posErr.append(None)
    else:
        corr_negErr.append(None)
        corr_posErr.append(None)

if all(corr is not None for corr in [corr_prePost_one, corr_prePost_both, corr_oneBoth_pre, corr_oneBoth_post]):
    # 提取相关性结果和置信区间
    corrs = [corr['r'].values[0] for corr in [corr_prePost_one, corr_prePost_both, corr_oneBoth_pre, corr_oneBoth_post] if corr is not None]
    corr_negErr = [corr['r'].values[0] - corr['CI95%'][0][0] for corr in [corr_prePost_one, corr_prePost_both, corr_oneBoth_pre, corr_oneBoth_post] if corr is not None]
    corr_posErr = [corr['CI95%'][0][1] - corr['r'].values[0] for corr in [corr_prePost_one, corr_prePost_both, corr_oneBoth_pre, corr_oneBoth_post] if corr is not None]

    summaryDict = {
                  'corrs': corrs, 'corr_negErr': corr_negErr, 'corr_posErr': corr_posErr,
                  'task': ['one-pleasure precued', 'one-pleasure postcued',
                           'combined-pleasure precued',
                           'combined-pleasure postcued']}
    summaryDf = pd.DataFrame(summaryDict)
    summaryDf.to_csv(dataDir + '/reliabilities_correlations.csv', index=False)

else:
    corrs = [corr['r'].values[0] for corr in [corr_prePost_one, corr_prePost_both, corr_oneBoth_pre, corr_oneBoth_post]
             if corr is not None]
    corr_negErr = [corr['r'].values[0] - corr['CI95%'][0][0] for corr in
                   [corr_prePost_one, corr_prePost_both, corr_oneBoth_pre, corr_oneBoth_post] if corr is not None]
    corr_posErr = [corr['CI95%'][0][1] - corr['r'].values[0] for corr in
                   [corr_prePost_one, corr_prePost_both, corr_oneBoth_pre, corr_oneBoth_post] if corr is not None]

    summaryDict = {'alpha': alphas,
                   'alpha_upperCI': alpha_upperCIs,
                  'alpha_lowerCI': alpha_lowerCIs,
                  'corrs': corrs,
                  'corr_negErr': corr_negErr,
                  'corr_posErr': corr_posErr,
                  'task': ['one-pleasure precued', 'one-pleasure postcued',
                           'combined-pleasure precued',
                           'combined-pleasure postcued']}
    summaryDf = pd.DataFrame(summaryDict)
    summaryDf.to_csv(dataDir + '/reliabilities_correlations.csv', index=False)