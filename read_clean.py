#LOAD LIBRARIES
import pandas as pd
import numpy as np
import os
import json
import urllib.request


#SET PATHS
train_path = 'https://raw.githubusercontent.com/asbiv/STAT6016_team_op/master/data/FinNum_training.json'
test_path = 'https://raw.githubusercontent.com/asbiv/STAT6016_team_op/master/data/FinNum_test.json'


#READ IN TRAIN AND TEST
#Train
with urllib.request.urlopen(train_path) as url:
    train_raw = json.loads(url.read().decode())

#Test
with urllib.request.urlopen(test_path) as url:
    test_raw = json.loads(url.read().decode())


#As dfs
train_df = pd.read_json(train_path)
test_df = pd.read_json(test_path)

train_df.dtypes

test_raw