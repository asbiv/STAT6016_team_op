import pandas as pd
from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence
import re
from sklearn import svm
from sklearn import metrics

# ==========================================================================================================
# create y.train
y_value = pd.DataFrame({'category-subcategory':["Monetary money", "Monetary quote", "Monetary change", 
                                                "Monetary buy price", "Monetary sell price", "Monetary forecast", 
                                                "Monetary stop loss", "Monetary support or resistance", 
                                                "Percentage relative", "Percentage absolute", "Option exercise price",
                                                "Option maturity date", "Indicator Indicator", "Temporal date",
                                                "Temporal time", "Quantity Quantity", "Product Number Product Number"],
                         'value':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]})
category = pd.DataFrame()
dupe_df['category'] = dupe_df['category'].map(lambda x: x[0] if type(x) == list else x)
dupe_df['subcategory'] = dupe_df['subcategory'].map(lambda x: x[0] if type(x) == list else x)
category = dupe_df.loc[:,'category'].map(str) + " " + dupe_df.loc[:,'subcategory'].map(str)
category = category.to_frame(name='category-subcategory')
category = category.merge(y_value, on='category-subcategory')
y_value = category.loc[:,'value']

# ===========================================================================================================
# create x.train

# Option 1 (higher accuracy)
x_value = pd.DataFrame(columns=["%","percent","pc","pct","up","down","decline","increase","growth","gain",
                                "lose","+","-","january","jan","february","feb","march","mar","april","apr",
                                "may","june","jun","july","jul","august","aug","september","sept","sep",
                                "october","oct","november","nov","december","dec","ma","dma","sma","ema",
                                "rsi","ichimoku","day","week","month","year","mos","yrs","second","sec",
                                "minute","min","mins","hour","hr","hrs","p.m.","pm","a.m.","call","put"],
                        index=range(300))

for i in range(0,300):
    for a in range(0,len(train_token[i])):
        for b in range(0,62):
            x_value.iloc[i,b] = 1 if train_token[i][a] == x_value.columns[b] else 0 


# Option 2 (lower accuracy)
#keys = {"key_p": ["%","percent","pc","pct"],
#        "key_r": ["up","down","decline","increase","growth","gain","lose","+","-"],
#        "key_m": ["january","jan","february","feb","march","mar","april","apr","may","june","jun","july","jul","august","aug","september","sept","sep","october","oct","november","nov","december","dec"],
#        "key_i": ["ma","dma","sma","ema","rsi","ichimoku"],
#        "key_d": ["day","week","month","year","mos","yrs","years", "wk", "weeks", "wks"],
#        "key_t": ["second","sec","minute","min","mins","hour","hr","hrs","p.m.","pm","a.m.", "a.m", "p.m"],
#        "key_callput": ["call","put"]}
#
#keynames = ["key_p","key_r","key_m","key_i","key_d","key_t", "key_callput"]
#
#key_vars = pd.DataFrame([], columns=keynames)
#
#def is_number(s):
#    try:
#        int(s)
#        return True
#    except ValueError:
#        return False
#
#def key_loop(s):
#    global key_vars
#    key_counts = pd.DataFrame(np.zeros((1,7)),columns=keynames)
#    for keylist in keys:
#        if len(list(set(s) & set(keys[keylist]))) > 0:
#            key_counts[keylist] = 1
#    key_vars = key_vars.append(key_counts.sum(), ignore_index=True)
#
#train_token_lower = train_token
#for i in range(0,6521):
#    train_token_lower[i] = [element.lower() for element in train_token[i]]
#train_token_lower.map(lambda x: key_loop(x))
#print(key_vars.head())
#x_value = key_vars

# build SVM
X_train, X_test, y_train, y_test = train_test_split(x_value, y_value[0:300], test_size=0.3)

clf = svm.SVC(kernel='linear') # Linear Kernel
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred) 
