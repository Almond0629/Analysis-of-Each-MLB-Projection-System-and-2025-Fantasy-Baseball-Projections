from pybaseball import *
import pandas as pd
import math
from statistics import mean
import numpy as np
import unicodedata
from unidecode import unidecode

'''
      batting_stats index:
      Season, Name, Team: 1~3
      G, AB, PA, H: 5~8, 2B: 10, HR: 12, AVG: 24, BB%: 35, K%: 36
      OBP, SLG, OPS, ISO, BABIP: 38~42
      LD%, GB%, FB%: 44~46
      
      Others:
      a = stats[0:15] is a Pandas Dataframe
      to_numpy() converts into a 2D Numpy array
'''
def find_player(name):
    name = name.split(' ', 1)
    player = playerid_lookup(name[1], name[0])
    if len(player) == 0: # check if player name is correct
        find_player = playerid_lookup(name[1], name[0], fuzzy=True)
        for j in range(len(find_player)):
            if find_player.iloc[j, 7] == 2022 or find_player.iloc[j, 7] == 2023 or find_player.iloc[j, 7] == 2024:
                return find_player.at[j, 'key_mlbam']
    if len(player) == 1:
        return player.at[0, 'key_mlbam']
    
def calculate_stats(df):
    df['OBP'] = (df['H'] + df['BB'] + df['HBP']) / (df['AB'] + df['BB'] + df['HBP'] + df['SF'])
    df['SLG'] = (df['1B'] + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / df['AB']
    df['OPS'] = df['OBP'] + df['SLG']
    df['BABIP'] = (df['H'] - df['HR']) / (df['AB'] - df['SO'] - df['HR'] + df['SF'])
    df['K%'] = df['SO'] / df['PA']
    df['BB%'] = df['BB'] / df['PA']
    df['K%'] = pd.to_numeric(df['K%'], errors='coerce') * 100
    df['BB%'] = pd.to_numeric(df['BB%'], errors='coerce') * 100
    return df

def scale_stats(df):
    columns_to_scale = df.columns.to_list()[4:22]   # ~NSB
    df['scaling_ratio'] = df['real_PA'] / df['PA']
    for col in columns_to_scale:
        df[col] = df[col] * df['scaling_ratio']
    df.drop(columns=['scaling_ratio'], inplace=True)
    return df

def format_name_to_fangraphs(name):
    if ',' in name:
        last, first = name.split(', ')
        return f'{first} {last}'
    return name

def normalize_name(name):
    return unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')


def spray_angle(x, y):
    angle = math.atan((x - 130) / (213 - y))  # Using an equation I found online to calculate the spray angle
    return math.degrees(angle)

def create_projection(df):
    atc = './Hitters/Fantasy/2025_ATC_Projections_Fantasy.csv'
    tb = './Hitters/Fantasy/2025_THE_BAT_Projections_Fantasy.csv'
    tbx = './Hitters/Fantasy/2025_THE_BAT_X_Projections_Fantasy.csv'
    dc = './Hitters/Fantasy/2025_DC_Projections_Fantasy.csv'
    zips = './Hitters/Fantasy/2025_ZIPS_Projections_Fantasy.csv'
    for proj in [atc, tb, tbx, dc, zips]:
        proj_df = pd.read_csv(proj)
        proj_df['HIP'] = proj_df['H'] - proj_df['HR']
        proj_df['1B%'] = proj_df['1B'] / proj_df['HIP']
        proj_df['2B%'] = proj_df['2B'] / proj_df['HIP']
        proj_df['3B%'] = proj_df['3B'] / proj_df['HIP']
        proj_df['HR%'] = proj_df['HR'] / proj_df['PA']
        proj_df['R%'] = proj_df['R'] / proj_df['PA']
        proj_df['RBI%'] = proj_df['RBI'] / proj_df['PA']
        proj_df['IBB%'] = proj_df['IBB'] / proj_df['PA']
        proj_df['HBP%'] = proj_df['HBP'] / proj_df['PA']
        proj_df['SF%'] = proj_df['SF'] / proj_df['PA']
        proj_df['SH%'] = proj_df['SH'] / proj_df['PA']
        proj_df['SB%'] = proj_df['SB'] / proj_df['PA']
        proj_df['CS%'] = proj_df['CS'] / proj_df['PA']
        proj_df.to_csv(proj, index=False)
    atc_df = pd.read_csv(atc)
    tb_df = pd.read_csv(tb)
    tbx_df = pd.read_csv(tbx)
    dc_df = pd.read_csv(dc)
    zips_df = pd.read_csv(zips)
    df['BABIP'] = tb_df['BABIP'] * 0.6 + tbx_df['BABIP'] * 0.4
    df['K%'] = dc_df['K%']
    df['BB%'] = tbx_df['BB%'] * 0.9 + dc_df['BB%'] * 0.1
    df['HR%'] = atc_df['HR%'] * 0.6 + tbx_df['HR%'] * 0.4
    df['1B%'] = tbx_df['1B%'] * 0.6 + atc_df['1B%'] * 0.4
    df['2B%'] = tbx_df['2B%']
    df['3B%'] = atc_df['3B%'] * 0.6 + tb_df['3B%'] * 0.4
    df['R%'] = dc_df['R%'] * 0.6 + tbx_df['R%'] * 0.4 
    df['RBI%'] = atc_df['RBI%']
    df['IBB%'] = atc_df['IBB%'] * 0.5 + tbx_df['IBB%'] * 0.4 + dc_df['IBB%'] * 0.1
    df['HBP%'] = tbx_df['HBP%'] * 0.5 + tb_df['HBP%'] * 0.4 + zips_df['HBP%'] * 0.1
    df['SF%'] = tb_df['SF%'] * 0.6 + atc_df['SF%'] * 0.4
    df['SH%'] = zips_df['SH%'] * 0.6 + atc_df['SH%'] * 0.4
    df['SB%'] = atc_df['SB%'] * 0.9 + dc_df['SB%'] * 0.1
    df['CS%'] = atc_df['CS%'] * 0.9 + dc_df['CS%'] * 0.1
    return df




player_list_path = './Hitters/Fantasy/2025_Fantasy_Hitters.csv'
player_list = pd.read_csv(player_list_path)
# projection_type = 'ZIPS'
# path = f'./Hitters/{projection_type}/2025_{projection_type}_Projections.csv'
# path_df = pd.read_csv(path)

# # Get rid of accents in names and get real life stats
# path_df['Name'] = path_df['Name'].apply(normalize_name)
# path_df.to_csv(path, index=False)

# path_df = calculate_stats(path_df)
# path_df.to_csv(path, index=False)

# names = set(player_list['Name'])
# projection_names = set(path_df['Name'])
# missing_names = names - projection_names

# if missing_names:
#     for name in missing_names:
#         print(name)

# filtered_df = path_df[path_df['Name'].isin(names)]
# filtered_df.to_csv(f'./Hitters/Fantasy/2025_{projection_type}_Projections_Fantasy.csv', index=False)

hitters_df = create_projection(player_list)
hitters_df.to_csv('./Hitters/Fantasy/2025_Fantasy_Hitters_modified.csv', index=False)

# print (find_player_name('guerrero jr.','vladimir'))
# á
# é
# í
# ó
# ú