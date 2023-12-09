# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 15:03:27 2021

@author: abrielmann

replicate the leave-one-out model fitting analyses for both-items-cued trials
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
# cue = 'Pre' # 'Pre' or 'Post
cue = 'Post'
#%% define cost functions for the 3 different models
def cost_linear(parameters, data):
    weight = 0.5
    a = parameters[0]
    b = parameters[1]

    imagePleasure = data.baselineImage
    musicPleasure = data.baselineMusic
    ratings = data.rating
    predictions = a+ b*(weight*imagePleasure + (1-weight)*musicPleasure)
    cost = np.sqrt(np.mean((predictions-ratings)**2))
    return cost

def cost_averaging(data):
    predictions = (data.baselineImage + data.baselineMusic)/2
    ratings = data.rating
    cost = np.sqrt(np.mean((predictions-ratings)**2))
    return cost

def cost_linear_modality(parameters, data):
    weight = parameters

    imagePleasure = data.baselineImage
    musicPleasure = data.baselineMusic
    ratings = data.rating
    predictions = weight*imagePleasure + (1-weight)*musicPleasure
    cost = np.sqrt(np.mean((predictions-ratings)**2))
    return cost

#%% settings for optimization
bounds_linear_modality = ((0, 1), ) # bounds for parameter fitting
startValue_linear_modality = [0.5]
bounds_compress = ((0, 10), (0, 1), ) # bounds for parameter fitting
startValue_compress = [0,1]
bounds_extreme = ((-10, 0), (1, 10), ) # bounds for parameter fitting
startValue_extreme = [0, 1]

#%% fetch pre-processed data
df = pd.read_csv(dataDir + '/all_data.csv')
# get some df properties we will need
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
# set up vvariables we want to record
avgRes_linear_modality = []
avgRmse_linear_modality = []
avgRes_compress = []
avgRmse_compress = []
avgRes_extreme = []
avgRmse_extreme = []
avgRmse_averaging = []

for peep in participants:
    peepDf = df.loc[(df['participant']==peep) & (df['cued'] =='Both') & (df['cueTime']==cue)]

    resList_linear_modality = []
    rmseList_linear_modality = []
    resList_compress = []
    rmseList_compress = []
    resList_extreme = []
    rmseList_extreme = []
    rmseList_averaging = []
    
    # We do LOOCV 'by hand', looping through all trials
    for trial in peepDf.index:
        train = peepDf[~peepDf.index.isin([trial])]
        test = peepDf[peepDf.index.isin([trial])]

        res_linear_modality = minimize(cost_linear_modality,
                                       startValue_linear_modality,
                                       args=(train,),
                    bounds=bounds_linear_modality,
                    options={'disp': False, 'maxiter': 1e5, 'ftol': 1e-08})
        resList_linear_modality.append(res_linear_modality.x)
        rmse_linear_modality = cost_linear_modality(res_linear_modality.x,
                                                    test)
        rmseList_linear_modality.append(res_linear_modality.fun)

        res_compress = minimize(cost_linear, startValue_compress, args=(train,),
                    bounds=bounds_compress,
                    options={'disp': False, 'maxiter': 1e5, 'ftol': 1e-08})
        resList_compress.append(res_compress.x)
        rmse_compress = cost_linear(res_compress.x, test)
        rmseList_compress.append(res_compress.fun)

        res_extreme = minimize(cost_linear, startValue_extreme, args=(train,),
                    bounds=bounds_extreme,
                    options={'disp': False, 'maxiter': 1e5, 'ftol': 1e-08})
        resList_extreme.append(res_extreme.x)
        rmse_extreme = cost_linear(res_extreme.x, test)
        rmseList_extreme.append(res_extreme.fun)

        rmse_averaging = cost_averaging(test)
        rmseList_averaging.append(rmse_averaging)

    avgRes_linear_modality.append(np.mean(resList_linear_modality))
    avgRmse_linear_modality.append(np.mean(rmseList_linear_modality))

    avgRes_compress.append(np.mean(resList_compress,axis=0))
    avgRmse_compress.append(np.mean(rmseList_compress))

    avgRes_extreme.append(np.mean(resList_extreme,axis=0))
    avgRmse_extreme.append(np.mean(rmseList_extreme))
    avgRmse_averaging.append(np.mean(rmseList_averaging))
# 假设 avgRes_compress 是一个列表的列表

# 确保 avgRes_compress 中的每个元素都是列表
avgRes_compress = [item if isinstance(item, list) or isinstance(item, np.ndarray) else [item] for item in avgRes_compress]

# 然后计算每个列表的长度
lengths = [len(sublist) for sublist in avgRes_compress]

# 你的后续代码...

if len(resList_compress) > 1:
    avgRes_compress.append(np.mean(resList_compress,axis=0))
else:
    # 处理 resList_compress 为空或只有一个元素的情况
    # 例如，如果列表为空，可以跳过此次迭代或添加一个默认值
    # 如果列表只有一个元素，则可以直接添加这个元素
    avgRes_compress.append(resList_compress[0] if resList_compress else 0)


# 现在你可以使用索引访问它
# result = avgRes_a_compress_array[:, 0]

# %% visualize
# from matplotlib import pyplot as plt
import seaborn as sns

# 确保 avgRes_compress 和 avgRes_extreme 都是二维数组
# 这里假设每个列表都是由两个数字组成的列表
avgRes_compress_array = np.array(avgRes_compress)
avgRes_extreme_array = np.array(avgRes_extreme)

# 确保它们是二维数组
if avgRes_compress_array.ndim == 2 and avgRes_compress_array.shape[1] == 2:
    avgRes_a_compress = avgRes_compress_array[:, 0]
    avgRes_b_compress = avgRes_compress_array[:, 1]
else:
    # 处理错误情况或调整代码
    avgRes_a_compress = avgRes_b_compress = np.array([np.nan] * len(participants))

if avgRes_extreme_array.ndim == 2 and avgRes_extreme_array.shape[1] == 2:
    avgRes_a_extreme = avgRes_extreme_array[:, 0]
    avgRes_b_extreme = avgRes_extreme_array[:, 1]
else:
    # 处理错误情况或调整代码
    avgRes_a_extreme = avgRes_b_extreme = np.array([np.nan] * len(participants))

# 创建 DataFrame
resDf = pd.DataFrame({
    'participant': participants,
    'avgRMSE_averaging': avgRmse_averaging,
    'avgRMSE_linear_modality': avgRmse_linear_modality,
    'avgRMSE_compress': avgRmse_compress,
    'avgRMSE_extreme': avgRmse_extreme,
    'avgRes_imageWeight_linear': avgRes_linear_modality,
    'avgRes_a_compress': avgRes_a_compress,
    'avgRes_b_compress': avgRes_b_compress,
    'avgRes_a_extreme': avgRes_a_extreme,
    'avgRes_b_extreme': avgRes_b_extreme
})

# 保存 DataFrame 到 CSV 文件
resDf.to_csv(os.path.join(dataDir, 'res_both_' + cue + 'cue.csv'), index=False)

