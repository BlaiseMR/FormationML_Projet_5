import bson
import json
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd

J_to_cal = 0.239006
cal_to_J = 1/J_to_cal

def check_type(df):
    for c in df.columns:
        print("Columns {} is type of {}" .format(c, type(df.loc[1][c])) )
        
def empty_columns(df):
    columns = []
    empty_columns = []
    df_count = df.count()
    
    for c in df.columns:
        if df_count.loc[c] > 0:
            columns.append(c)
            print("Column {} have {} entries -> {}" .format(c,df_count.loc[c],'not empty'))
        else:
            empty_columns.append(c)
            print("Colomn {} have {} entries -> {}" .format(c,df_count.loc[c],'empty'))
    return empty_columns, columns

def notfull_columns(df, ratio):
    n_rows = len(df)
    columns = []
    half_empty_columns = []
    df_count = df.count()

    for c in df.columns:
        #print(c)
        if df_count.loc[c] > n_rows*ratio:
            columns.append(c)
        else:
            half_empty_columns.append(c)
    
    return half_empty_columns, columns

def Inspect_rows(df, cat, cat_id):   
    result = df[df[cat] == cat_id]
    return result

def check_duplicates(df, colonnes):
    result = df[df.duplicated(subset=colonnes, keep=False)].sort_values(colonnes)
    return result

def process_duplicates(df):
    result = df.drop_duplicates(subset=['code'])
    
    result = pd.concat([ 
        result[result['product_name'].isna()],
        result[result['product_name'].isnull()], 
        result.drop_duplicates( subset=['product_name'] )
    ])
    return result

def nutriscore_point(amount, cat):
    if cat == 'Energy':
        if amount <= 335 :
            return 0
        elif amount < 670 :
            return 1
        elif amount < 1005 :
            return 2
        elif amount < 1340 :
            return 3
        elif amount < 1675 :
            return 4
        elif amount < 2010 :
            return 5
        elif amount < 2345 :
            return 6
        elif amount < 2680 :
            return 7
        elif amount < 3015 :
            return 8
        elif amount < 3350 :
            return 9
        else :
            return 10
        
    if cat == 'Fats':
        if amount <= 1 :
            return 0
        elif amount < 1 :
            return 1
        elif amount < 2 :
            return 2
        elif amount < 3 :
            return 3
        elif amount < 4 :
            return 4
        elif amount < 5 :
            return 5
        elif amount < 6 :
            return 6
        elif amount < 7 :
            return 7
        elif amount < 8 :
            return 8
        elif amount < 9 :
            return 9
        else :
            return 10
        
    if cat == 'Sugars':
        if amount <= 4.5 :
            return 0
        elif amount < 9 :
            return 1
        elif amount < 13.5 :
            return 2
        elif amount < 18 :
            return 3
        elif amount < 22.5 :
            return 4
        elif amount < 27 :
            return 5
        elif amount < 30 :
            return 6
        elif amount < 36 :
            return 7
        elif amount < 40 :
            return 8
        elif amount < 45 :
            return 9
        else :
            return 10
        
    if cat == 'Sodium':
        if amount <= 90 :
            return 0
        elif amount < 180 :
            return 1
        elif amount < 270 :
            return 2
        elif amount < 360 :
            return 3
        elif amount < 450 :
            return 4
        elif amount < 540 :
            return 5
        elif amount < 630 :
            return 6
        elif amount < 720 :
            return 7
        elif amount < 810 :
            return 8
        elif amount < 900 :
            return 9
        else :
            return 10
        
    if cat == 'Fruits':
        if np.isnan(amount):
            return 0
        elif amount <= 40 :
            return 0
        elif amount < 60 :
            return 1
        elif amount < 80 :
            return 2
        else:
            return 5
        
    if cat == 'Fibers':
        if np.isnan(amount):
            return 0
        elif amount <= 0.9 :
            return 0
        elif amount < 1.9 :
            return 1
        elif amount < 2.8 :
            return 2
        elif amount < 3.7 :
            return 3
        elif amount < 4.7 :
            return 4
        else :
            return 5
        
    if cat == 'Proteins':
        if np.isnan(amount):
            return 0
        elif amount <= 1.6:
            return 0
        elif amount < 3.2 :
            return 1
        elif amount < 4.8 :
            return 2
        elif amount < 6.4 :
            return 3
        elif amount < 8.0 :
            return 4
        else :
            return 5
        
def compute_nutrigrade(score):
    if score < 0 :
        return 'a'
    elif score < 3 :
        return 'b'
    elif score < 11 :
        return 'c'
    elif score < 19 :
        return 'd'
    else :
        return 'e'

def complete_nutriscore(row):
    N = 0
    P = 0
    if np.isnan(row['nutriscore_score']) :
        
        # Negative impact 
            # Energy density
        if np.isnan(row['energy-kj_100g']) == False and row['energy-kj_100g'] == 0:
            N += nutriscore_point(row['energy-kj_100g'], 'Energy')
        else:
            #print("Fail energy :", row['energy-kj_100g'] )
            N += 10
        
            # Satured fats
        if np.isnan(row['saturated-fat_100g']) == False :
            N += nutriscore_point(row['saturated-fat_100g'], 'Fats')
        else:
            #print("Fail fats :", row['saturated-fat_100g'] )
            N += 10
        
            # Sugars
        if np.isnan(row['sugars_100g']) == False :
            N += nutriscore_point(row['sugars_100g'], 'Sugars')
        else:
            #print("Fail sugars", row['sugars_100g'] )
            N += 10
        
            # Sodium
        if np.isnan(row['sodium_100g']) == False :
            N += nutriscore_point(row['sodium_100g'], 'Sodium')
        else:
            #print("Fail sodium", row['sodium_100g'] )
            N += 10

        # Positive impact
            # Percent of vegetable/legumes
        if np.isnan(row['fruits-vegetables-nuts_100g']) == False :
            P += nutriscore_point(row['fruits-vegetables-nuts_100g'], 'Fruits')

            # Fibers
        if np.isnan(row['fiber_100g']) == False :
            P += nutriscore_point(row['fiber_100g'], 'Fibers')

            # Proteins
        if np.isnan(row['proteins_100g']) == False :
            P += nutriscore_point(row['proteins_100g'], 'Proteins')

        row['nutriscore_score'] = N - P
        row['nutriscore_grade'] = compute_nutrigrade(N-P)
        #print(row['code'], ':', row['nutriscore_score'], '(', N, P, ')', row['nutriscore_grade'])
        #print(row['energy-kj_100g'], row['saturated-fat_100g'], row['sugars_100g'], row['sodium_100g'])
        #print(row['fruits-vegetables-nuts_100g'], row['fiber_100g'], row['proteins_100g'])
    return row

def complete_energy(df):
    if np.isnan(df['energy-kj_100g']) and np.isnan(df['energy-kcal_100g']) == False :
        check = df['energy-kj_100g']
        df['energy-kj_100g'] = df['energy-kcal_100g']*cal_to_J
        #print(df['code'], ' )', check, '->',df['energy-kcal_100g']*cal_to_J, ':', df['energy-kcal_100g'])
    return df

def check_filling(df):
    n_rows = len(df)
    columns = []
    ratio = []
    df_count = df.count()

    for c in df.columns:
        ratio.append(df_count.loc[c] / n_rows * 100)
        columns.append(c)
    
    df2 = pd.DataFrame({'Ratio':ratio, 'columns':columns})
    df2 = df2.sort_values(by = 'Ratio', axis = 0, ascending = False)
    
    ax = df2.plot.bar(x='columns', y='Ratio', rot=90, figsize=(16,6))
    plt.xlabel('Columns')
    plt.ylabel('% de remplissage')
    
    return df2

def check_outliers(df, field, v_min, v_max):
    if df[field] < v_min or df[field] > v_max :
        df[field] = np.nan
    return df