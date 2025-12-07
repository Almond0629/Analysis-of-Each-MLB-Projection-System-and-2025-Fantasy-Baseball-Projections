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
    atc = './Projection_model/Pitchers/Fantasy/2025_ATC_Projections_Fantasy.csv'
    # tb = './Projection_model/Pitchers/Fantasy/2025_THE_BAT_Projections_Fantasy.csv'
    # tbx = './Projection_model/Pitchers/Fantasy/2025_THE_BAT_X_Projections_Fantasy.csv'
    # dc = './Projection_model/Pitchers/Fantasy/2025_DC_Projections_Fantasy.csv'
    # zips = './Projection_model/Pitchers/Fantasy/2025_ZIPS_Projections_Fantasy.csv'
    oopsy = './Projection_model/Pitchers/Fantasy/2025_OOPSY_Projections_Fantasy.csv'
    for proj in [atc, oopsy]:
        proj_df = pd.read_csv(proj)
        proj_df['W%'] = proj_df['W'] / proj_df['G']
        proj_df['L%'] = proj_df['L'] / proj_df['G']
        proj_df['QS%'] = proj_df['QS'] / proj_df['G']
        proj_df['SV%'] = proj_df['SV'] / proj_df['G']
        proj_df['HLD%'] = proj_df['HLD'] / proj_df['G']
        proj_df['BS%'] = proj_df['BS'] / proj_df['G']
        proj_df['H%'] = proj_df['H'] / proj_df['TBF']
        proj_df['R%'] = proj_df['R'] / proj_df['TBF']
        proj_df['ER%'] = proj_df['ER'] / proj_df['TBF']
        proj_df['HR%'] = proj_df['HR'] / proj_df['TBF']
        proj_df['IBB%'] = proj_df['IBB'] / proj_df['TBF']
        proj_df['HBP%'] = proj_df['HBP'] / proj_df['TBF']
        proj_df.to_csv(proj, index=False)
    atc_df = pd.read_csv(atc)
    # tb_df = pd.read_csv(tb)
    # tbx_df = pd.read_csv(tbx)
    # dc_df = pd.read_csv(dc)
    # zips_df = pd.read_csv(zips)
    oopsy_df = pd.read_csv(oopsy)
    df['W%'] = atc_df['W%'] * 0.7 + oopsy_df['W%'] * 0.3
    df['L%'] = atc_df['L%'] * 0.7 + oopsy_df['L%'] * 0.3
    df['QS%'] = atc_df['QS%'] * 0.7 + oopsy_df['QS%'] * 0.3
    df['SV%'] = atc_df['SV%'] * 0.7 + oopsy_df['SV%'] * 0.3
    df['HLD%'] = atc_df['HLD%'] * 0.7 + oopsy_df['HLD%'] * 0.3
    df['BS%'] = atc_df['BS%']
    df['H%'] = oopsy_df['H%'] * 0.7 + atc_df['H%'] * 0.3
    df['R%'] = oopsy_df['R%'] * 0.7 + atc_df['R%'] * 0.3
    df['ER%'] = oopsy_df['ER%'] * 0.7 + atc_df['ER%'] * 0.3
    df['HR%'] = oopsy_df['HR%'] * 0.7 + atc_df['HR%'] * 0.3
    df['IBB%'] = atc_df['IBB%']
    df['HBP%'] = atc_df['HBP%'] * 0.7 + oopsy_df['HBP%'] * 0.3
    df['K%'] = oopsy_df['K%'] * 0.7 + atc_df['K%'] * 0.3
    df['BB%'] = oopsy_df['BB%'] * 0.7 + atc_df['BB%'] * 0.3
    return df




player_list_path = './Projection_model/Pitchers/Fantasy/2025_Fantasy_Starters.csv'
player_list = pd.read_csv(player_list_path)
# projection_type = 'ATC'
# path = f'./Projection_model/Pitchers/{projection_type}/2025_{projection_type}_Projections.csv'
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
# filtered_df.to_csv(f'./Projection_model/Pitchers/Fantasy/2025_{projection_type}_Projections_Fantasy.csv', index=False)

pitchers_df = create_projection(player_list)
pitchers_df.to_csv('./Projection_model/Pitchers/Fantasy/2025_Fantasy_Starters.csv', index=False)

# print (find_player_name('guerrero jr.','vladimir'))
# á
# é
# í
# ó
# ú