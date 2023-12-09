# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 15:03:27 2021

@author: abrielmann

replicate the leave-one-out model fitting analyses for one-item-cued trials
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# 设置目录
homeDir = 'D:\\临时\\image_and_music_pleasure-main\\image_and_music_pleasure-main\\data'
experiment = 'exp4'
dataDir = os.path.join(homeDir, experiment)

# Do you want to look at pre or postcued?
cue = 'Post'
# cue = 'Pre'
#%% define cost functions for the 4 different models
def cost_linear(parameters, data):
    weight = parameters
    targetPleasure = data.baselineTarget
    distractorPleasure = data.baselineDistractor
    ratings = data.rating
    predictions = weight*targetPleasure + (1-weight)*distractorPleasure
    cost = np.sqrt(np.mean((predictions-ratings)**2))
    return cost

def cost_linear_modality(parameters, data):
    # modality dependent target weights
    weightImage = parameters[0]
    weightMusic = parameters[1]
    weight = np.repeat(weightImage, data.shape[0])
    weight[data.cued=='Music'] = weightMusic
    oneMinusWeight = [1-w for w in weight]
    targetPleasure = data.baselineTarget
    distractorPleasure = data.baselineDistractor
    ratings = data.rating
    predictions = weight*targetPleasure + oneMinusWeight*distractorPleasure
    cost = np.sqrt(np.mean((predictions-ratings)**2))
    return cost

def cost_accurate(data):
    predictions = data.baselineTarget
    ratings = data.rating
    cost = np.sqrt(np.mean((predictions-ratings)**2))
    return cost

def cost_highAtt(parameters, data):
    P_beau=parameters
    targetPleasure = data.baselineTarget.values
    distractorPleasure = data.baselineDistractor.values
    ratings = data.rating.values
    predictions = []
    for trial in range(len(targetPleasure)):
       if targetPleasure[trial]<P_beau:
           predictions.append(targetPleasure[trial])
       else:
           pred = (P_beau +
                   (distractorPleasure[trial] / targetPleasure[trial])
                   *(targetPleasure[trial] - P_beau))
           predictions.append(pred)
    cost = np.sqrt(np.mean((predictions-ratings)**2))
    return float(cost)

#%% settings for optimization
bounds_linear = ((0,1), ) # bounds for parameter fitting
startValue_linear = 0.5
bounds_linear_modality = ((0,1), (0,1),) # bounds for parameter fitting
startValue_linear_modality = [0.5, 0.5]
bounds_highAtt = ((1,9), ) # bounds for parameter fitting
startValue_highAtt = 5

#%% fetch pre-processed data
df = pd.read_csv(dataDir + '/all_data.csv')
# fetch some data properties we will need later on
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

#%% Loop through the individual participants
# first set up all variables we want to record
avgRes_linear = []
avgRmse_linear = []
avgRes_linear_modality = []
avgRmse_linear_modality = []
avgRes_highAtt = []
avgRmse_highAtt = []
avgRmse_accurate = []
avgRmse_averaging = []

for peep in participants:
    peepDf = df.loc[(df['participant']==peep) & (df['cued'] !='Both') & (df['cueTime']==cue)]

    resList_linear = []
    rmseList_linear = []
    resList_linear_modality = []
    rmseList_linear_modality = []
    resList_highAtt = []
    rmseList_highAtt = []
    rmseList_accurate = []
    rmseList_averaging = []
    
    # We do LOOCV 'by hand', looping through all trials
    for trial in peepDf.index:
        train = peepDf[~peepDf.index.isin([trial])]
        test = peepDf[peepDf.index.isin([trial])]

        res_linear = minimize(cost_linear, startValue_linear, args=(train,),
                    bounds=bounds_linear,
                    options={'disp': False, 'maxiter': 1e5, 'ftol': 1e-08})
        resList_linear.append(res_linear.x)
        rmse_linear = cost_linear(res_linear.x, test)
        rmseList_linear.append(res_linear.fun)

        res_linear_modality = minimize(cost_linear_modality,
                                       startValue_linear_modality, args=(train,),
                    bounds=bounds_linear_modality,
                    options={'disp': False, 'maxiter': 1e5, 'ftol': 1e-08})
        resList_linear_modality.append(res_linear_modality.x)
        rmse_linear_modality = cost_linear_modality(res_linear_modality.x, test)
        rmseList_linear_modality.append(res_linear_modality.fun)

        res_highAtt = minimize(cost_highAtt, startValue_highAtt, args=(train,),
                    bounds=bounds_highAtt,
                    options={'disp': False, 'maxiter': 1e5, 'ftol': 1e-08})
        resList_highAtt.append(res_highAtt.x)
        rmse_highAtt = cost_linear(res_highAtt.x, test)
        rmseList_highAtt.append(res_highAtt.fun)

        rmse_accurate = cost_accurate(test)
        rmse_averaging = cost_linear(0.5, test)
        rmseList_accurate.append(rmse_accurate)
        rmseList_averaging.append(rmse_averaging)

    avgRes_linear.append(np.mean(resList_linear))
    avgRmse_linear.append(np.mean(rmseList_linear))

    avgRes_linear_modality.append(np.mean(resList_linear_modality, axis=0))
    avgRmse_linear_modality.append(np.mean(rmseList_linear_modality))

    avgRes_highAtt.append(np.mean(resList_highAtt))
    avgRmse_highAtt.append(np.mean(rmseList_highAtt))
    avgRmse_accurate.append(np.mean(rmseList_accurate))
    avgRmse_averaging.append(np.mean(rmseList_averaging))



# 创建 DataFrame
resDf = pd.DataFrame({
    'participant': participants,
    'avgRes_linear': avgRes_linear,
    'avgRmse_linear': avgRmse_linear,
    'avgRes_linear_modality': avgRes_linear_modality,
    'avgRmse_linear_modality': avgRmse_linear_modality,
    'avgRes_highAtt': avgRes_highAtt,
    'avgRmse_highAtt': avgRmse_highAtt,
    'avgRmse_accurate': avgRmse_accurate,
    'avgRmse_averaging': avgRmse_averaging
})

# 将 DataFrame 保存为 CSV 文件
resDf.to_csv(os.path.join(dataDir, 'res_one_' + cue + 'cue.csv'), index=False)

