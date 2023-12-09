#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 18:06:34 2021

Last modified Wed May 18 2022: Cleaning up

@author: aennebrielmann

Get the descriptive distribution of initial ratings (per participant)
per modality
"""
import os
import pandas as pd


# 设置目录
homeDir = 'D:\\临时\\image_and_music_pleasure-main\\image_and_music_pleasure-main\\data'
experiment = 'exp4'
dataDir = os.path.join(homeDir, experiment)

#%% fetch pre-processed data
df = pd.read_csv(dataDir + '/results_per_participant.csv')

# #%% overview of demographics
# print(df[['Ed', 'Age', 'Race', 'YrsMusic', 'YrsArt',
#           'BMRQ_sum', 'AReA_total']].describe())
# print('\n')
# print(df['Sex'].value_counts())