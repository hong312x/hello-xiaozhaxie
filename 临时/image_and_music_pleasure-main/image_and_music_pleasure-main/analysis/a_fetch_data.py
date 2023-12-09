import os
import numpy as np
import pandas as pd

# 设置目录
homeDir = 'D:\\临时\\image_and_music_pleasure-main\\image_and_music_pleasure-main\\data'
experiment = 'exp4'
dataDir = os.path.join(homeDir, experiment)

# 读取原始数据
mainDf = pd.read_csv(os.path.join(dataDir, 'all_task.csv'))
baseDf = pd.read_csv(os.path.join(dataDir, 'all_baseline.csv'))
demoDf = pd.read_csv(os.path.join(dataDir, 'all_questionnaires.csv'))

# 从原始数据中创建一个新的 DataFrame
df = pd.DataFrame({'participant': mainDf['Prolific_ID'],
                   'trialPerBlock': mainDf['Trial.Number'],
                   'runningTrialNumber': mainDf.index,
                   'rating': mainDf['Response'],
                   'rt': 1000,  # 将所有值设置为 1000
                   'cueTime': mainDf['Display'],
                   'image': mainDf['ImageStim'],
                   'music': mainDf['MusicStim'],
                   'cued': mainDf['StimCued']})


# 设置用于构造 df 的空变量
baselineImage = []
baselineMusic = []
baselineTarget = []
baselineDistractor = []
cuedImage = []
cuedMusic = []

# 填充上述列表
for trial in df.index:
    peep = df.participant[trial]
    cue = df.cued[trial]
    image = df.image[trial]
    music = df.music[trial]

    imInd = (baseDf['Prolific_ID'] == peep) & (baseDf['stimulus'] == image)
    musicInd = (baseDf['Prolific_ID'] == peep) & (baseDf['stimulus'] == music)

    # 对于 baselineImage 和 baselineMusic
    if not baseDf.loc[imInd, 'Response'].empty:
        baselineImage.append(baseDf.loc[imInd, 'Response'].mean())
    else:
        baselineImage.append(np.nan)  # 或者其他合适的默认值

    if not baseDf.loc[musicInd, 'Response'].empty:
        baselineMusic.append(baseDf.loc[musicInd, 'Response'].mean())
    else:
        baselineMusic.append(np.nan)  # 或者其他合适的默认值

    # 对于 baselineTarget 和 baselineDistractor
    if cue == "Image":
        baselineTarget.append(baseDf.loc[imInd, 'Response'].mean() if not baseDf.loc[imInd, 'Response'].empty else np.nan)
        baselineDistractor.append(baseDf.loc[musicInd, 'Response'].mean() if not baseDf.loc[musicInd, 'Response'].empty else np.nan)
        cuedImage.append(image)
        cuedMusic.append(np.nan)
    elif cue == "Music":
        baselineTarget.append(baseDf.loc[musicInd, 'Response'].mean() if not baseDf.loc[musicInd, 'Response'].empty else np.nan)
        baselineDistractor.append(baseDf.loc[imInd, 'Response'].mean() if not baseDf.loc[imInd, 'Response'].empty else np.nan)
        cuedImage.append(np.nan)
        cuedMusic.append(music)
    else:
        baselineTarget.append(np.nan)
        baselineDistractor.append(np.nan)
        cuedImage.append(image)
        cuedMusic.append(music)


# 将新变量添加到 df 中
df['baselineImage'] = baselineImage
df['baselineMusic'] = baselineMusic
df['baselineTarget'] = baselineTarget
df['baselineDistractor'] = baselineDistractor
df['cuedImage'] = cuedImage
df['cuedMusic'] = cuedMusic

# 创建额外的新变量
df['baselinesMean'] = df[['baselineImage', 'baselineMusic']].mean(axis=1)
df['predRating'] = df['baselineTarget'].copy()
df.loc[df['cued'] == 'Both', 'predRating'] = df.loc[df['cued'] == 'Both', 'baselinesMean']
df['diffRatePred'] = df['rating'] - df['predRating']

# 排除特定参与者
if experiment == 'exp1_sonaShort':
    excluded = [4743, 4832, 4787, 4865, 4798]
elif experiment == 'exp2_sonaLong':
    excluded = [4828, 4767, 4746, 4855, 4757, 4780, 4882, 4737, 4800]
elif experiment == 'exp3_prolificShort':
    excluded = ['5df1fbdb99b2820fcfa3fa37']
else:


    # 保存结果为新的 csv 文件
    df.to_csv(os.path.join(dataDir, 'all_data.csv'), index=False)

# df = df[~df['participant'].isin(excluded)]
#
# # 保存结果为新的 csv 文件
# df.to_csv(os.path.join(dataDir, 'all_data.csv'), index=False)
