#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 09:41:57 2018

@author: chungchi
"""

consumer_key = "436288ce8cc12233"
consumer_secret = "5d4ce01992a2ee99b5fc4ae5c6a4aac53a290da6"
redirect_url = "http://saundersetc.com"
code = "39a60843230fb1ea3a39b077c24df019ad16d61f"

token_url = "https://api.stocktwits.com/api/2/oauth/token?client_id=" + consumer_key + "&client_secret=" + consumer_secret + "&code=" + code + "&grant_type=authorization_code&redirect_uri=" + redirect_url

import json
import time
import requests
import urllib.request
import os
os.chdir('/Users/Ab/Desktop/SYS6016_Local/team_op/phase_2/data')

token_info = requests.post(token_url)
token = json.loads(token_info.content.decode("utf8"))["access_token"]

raw_list = [
        #'https://raw.githubusercontent.com/asbiv/STAT6016_team_op/master/data/FinNum_training.json',
#        'https://raw.githubusercontent.com/asbiv/STAT6016_team_op/master/data/FinNum_test.json'#,
        'https://raw.githubusercontent.com/asbiv/STAT6016_team_op/master/data/FinNum_dev.json'
        ]

raw_local = [
        "FinNum_training","FinNum_test","FinNum_dev"
        ]

for FinNum in raw_list: #["FinNum_training","FinNum_dev","FinNum_test"]:
    with urllib.request.urlopen(FinNum) as url:
        data = json.loads(url.read().decode())
    #with open(FinNum + ".json") as f:
    #    data = json.load(f)
    
    #Authenticated calls are permitted 400 requests per hour and measured against the access token used in the request.
    not_found = []
    twt = []
    idx = []
    
    i = 0
    while(i != len(data)):
        print(i)
        if(data[i]["idx"] in idx):
            j = idx.index(data[i]["idx"])
            data[i]["tweet"] = twt[j]
            i = i + 1
            continue
        
        url = "https://api.stocktwits.com/api/2/messages/show/" + str(data[i]["id"]) + ".json?access_token=" + token
        tweet_info = json.loads(requests.get(url).content.decode("utf8"))
    
        if(tweet_info["response"]["status"] == 200):
            tweet = tweet_info["message"]["body"]
            data[i]["tweet"] = tweet
            twt.append(tweet)
            idx.append(data[i]["idx"])
            i = i + 1
        elif(tweet_info["response"]["status"] == 429):
            print("sleep one hour----from " + time.ctime() )
            time.sleep(3600)
        else:
            not_found.append(i)
            print(i)
            print(tweet_info)
            i = i + 1
            
    for i in not_found[::-1]:
        del data[i]
    
    print("Missing data: " + str(len(not_found)))
    print("Total: " + str(len(data)) + " instances")
    
    json.dump(data, open("FinNum_dev" + "_rebuilded.json", 'w'))