"""
Summary:
Evaluate the zone entry and exit abilities of players by charting players' performance on different metrics calculated using the outputs of RNN model.
"""
%reset -sf
import pandas as pd
import numpy as np
import re
import os
import copy
import matplotlib.pyplot as plt

# get shift data to look at avg TOI per game across players
shifts = pd.read_csv(os.path.join(os.getcwd(), 'import/BDC_2024_Womens_Shifts.csv'))
# calculate time on ice over the four games of data
game_toi = shifts.\
    assign(shift_minutes = lambda x: x.shift_length.str.extract(r'(\d{1,2})\:.+$').astype('int'),
    shift_seconds = lambda x: x.shift_length.str.extract(r'.+\:(\d{2}$)').astype('int'),
    toi_in_min = lambda x: (x.shift_minutes) + (x.shift_seconds/60)).\
    groupby(['team_name', 'player_name'], as_index = False)["toi_in_min"].\
        apply(lambda x : x.sum()).\
    sort_values(by=['team_name', 'toi_in_min'], ascending = [True, False])

# import play by play data with zone entry probabilities
pre_entry_with_probs = pd.read_csv(os.path.join(os.getcwd(), 'datasets/playbyplay_data_with_zone_entry_probabilities.csv'))

# Make chart to show players' contributions to zone exits and neutral zone play -------------------------------------------------------
flag_summary = copy.copy(pre_entry_with_probs)
flag_summary = flag_summary[flag_summary.\
            apply(lambda x: pd.notna(x['prob_success_entry']) &\
                  (x['active_team_sich']=="Even Strength"), 
                     axis=1)]

# flag high value plays
# puck carries that increase probability of valuable zone entry by more than 4%
# passes that increase probability of valuable zone entry by more than 10%
flag_summary['stick_value_gr_.04'] = np.select([(flag_summary['Event'] == 'Stick Handle') & \
             (flag_summary['vert_move']>0) & (flag_summary['prob_change'] >= 0.04)], [1], default=0)

flag_summary['pass_value_gr_.1'] = np.select([(flag_summary['Event'] == 'Play') & \
             (flag_summary['vert_move']>0) & (flag_summary['prob_change'] >= 0.1)], [1], default=0)    

# calculate how many high value plays each players had per min of TOI
player_level_summary_temp = flag_summary.\
    groupby(['Player'], as_index = False)[
                                          "stick_value_gr_.04",
                                          'pass_value_gr_.1'
                                          ].\
        apply(lambda x : x.sum()).\
        merge(game_toi, 
        left_on = ['Player'],right_on = ['player_name'],
          how = 'left')

player_level_summary_per_game = copy.copy(player_level_summary_temp)
player_level_summary_per_game[[
                                      "stick_value_gr_.04",
                                      'pass_value_gr_.1']] = \
    player_level_summary_per_game[[ 
                                          "stick_value_gr_.04",
                                          'pass_value_gr_.1'
                                          ]].\
        div(player_level_summary_per_game.toi_in_min, axis=0)
        
player_level_summary_per_game = player_level_summary_per_game.\
    drop(columns = ["team_name","player_name", "toi_in_min"])

player_level_summary = player_level_summary_temp.\
        merge(player_level_summary_per_game, 
        left_on = ['Player'],right_on = ['Player'],
          how = 'left', suffixes=('', '_per_min')).\
            drop(columns = ["player_name"])
            
    
# zone exit scatter plot by player: x axis shows incidence of high value puck carries, y axis shows incidence of high value passes
# scatter plots only include players that have non zero values for both metrics 
#scatter plots only include players that have over 30 min TOI over 4 games
zone_exit_plotting = player_level_summary[player_level_summary.\
    apply(lambda x: (x["stick_value_gr_.04_per_min"]>0) & \
          (x['pass_value_gr_.1']>0) & (x['toi_in_min']>30), axis=1)]

x_exit = copy.copy(zone_exit_plotting[['stick_value_gr_.04_per_min']]).to_numpy().reshape(-1)
y_exit = copy.copy(zone_exit_plotting[['pass_value_gr_.1_per_min']]).to_numpy().reshape(-1)
team_exit = copy.copy(zone_exit_plotting[['team_name']]).to_numpy().reshape(-1)
player_exit = copy.copy(zone_exit_plotting[['Player']]).to_numpy().reshape(-1)
color_exit = np.where(team_exit == 'Women - Canada', "red", "blue")

fig_exit = plt.figure(figsize=(20,8))
ax_exit = fig_exit.add_subplot(111)
plt.scatter(x_exit, y_exit, c = color_exit, s = 50)
plt.title("High-Value Passes and Puck Carries in Breakout and\nNeutral Zone at Even Strength", 
          fontsize = 35)
plt.xlabel("Impactful Puck Carries, Per Minute of TOI",fontsize = 20)
plt.ylabel("Impactful Passes, Per Minute of TOI",fontsize = 20)
for i, txt in enumerate(player_exit):
    if txt == "Ashton Bell":
        ax_exit.text(x_exit[i], y_exit[i], txt, fontsize = 15, horizontalalignment = 'right',
                verticalalignment = 'bottom')
        continue;
    elif txt == "Brianne Jenner":
        ax_exit.text(x_exit[i], y_exit[i], txt, fontsize = 15, horizontalalignment = 'right',
                verticalalignment = 'bottom')
        continue;
    elif txt == "Blayre Turnbull":
        ax_exit.text(x_exit[i], y_exit[i], txt, fontsize = 15, horizontalalignment = 'right',
                verticalalignment = 'bottom')
        continue;
    elif txt == "Sarah Nurse":
        ax_exit.text(x_exit[i], y_exit[i], txt, fontsize = 15, horizontalalignment = 'right',
                verticalalignment = 'bottom')
        continue;
    ax_exit.text(x_exit[i], y_exit[i], txt, fontsize = 15, horizontalalignment = 'left',
            verticalalignment = 'bottom')
    
plt.savefig("charts/zone_exit_chart.png", format="png", bbox_inches='tight')
    
# Make chart to show players' contributions to zone entries --------------------------------------------------------------------
# first get all zone entries data and the final probability of having a valuable zone entry before it happens

last_prob = pre_entry_with_probs.\
    assign(last_grp_order_before_zone_entry = \
    pre_entry_with_probs.groupby(['unique_poss_id'], as_index=False)['grp_order'].transform('max'))

last_prob = last_prob[last_prob.\
     apply(lambda x: (x['last_grp_order_before_zone_entry']==x['grp_order']), axis=1)]
last_prob_forjoin = copy.copy(last_prob[['unique_poss_id', 'prob_success_entry']])

# import play by play data that contains all zone entries
with_all_zone_entries = pd.read_csv(os.path.join(os.getcwd(), 'datasets/playbyplay_data_with_zone_entries.csv'))

zone_entries = with_all_zone_entries[with_all_zone_entries.\
    apply(lambda x: (x['Event']=="Zone Entry"), axis=1)].\
    merge(last_prob_forjoin, 
    on = ['unique_poss_id'],
      how = 'left')
    
zone_entries = copy.copy(zone_entries[['unique_poss_id',
    'Team', 'Player', 'Detail_1', 'active_team_sich', 'shot_flag_poss_level',
    'prob_success_entry','clean_success_entry_poss_level'
    ]])

# filter to even strength
zone_entries = copy.copy(zone_entries[zone_entries.\
     apply(lambda x: pd.notna(x['prob_success_entry']) &\
           (x['active_team_sich']=="Even Strength") &\
              # (x['Detail_1']!="Dumped"), 
              (x['Detail_1']=="Carried"), 
              axis=1)])

# flag if probability of success is less than 20%
zone_entries['succ_entry_lt_.2'] = np.select([(zone_entries['prob_success_entry']<.2) &\
            (zone_entries['clean_success_entry_poss_level'] == 1)], [1], default=0)

# calculate how many valuable zone entries, and how many valuable and difficult (<20% success probability) zone entries per min of TOI each player has
player_entry_summ_temp = zone_entries.\
    groupby(['Player'], as_index = False)["clean_success_entry_poss_level","succ_entry_lt_.2"].\
        apply(lambda x : x.sum()).\
        merge(game_toi, 
        left_on = ['Player'],right_on = ['player_name'],
          how = 'left')

player_entry_summ_per_game = copy.copy(player_entry_summ_temp)
player_entry_summ_per_game[["clean_success_entry_poss_level",
                                      "succ_entry_lt_.2"]] = \
    player_entry_summ_per_game[["clean_success_entry_poss_level",
                                          "succ_entry_lt_.2"]].\
        div(player_entry_summ_per_game.toi_in_min, axis=0)
        
player_entry_summ_per_game = player_entry_summ_per_game.\
    drop(columns = ["team_name","player_name", "toi_in_min"])

player_entry_summ = player_entry_summ_temp.\
        merge(player_entry_summ_per_game, 
        left_on = ['Player'],right_on = ['Player'],
          how = 'left', suffixes=('', '_per_min')).\
            drop(columns = ["player_name"])



# zone entry charts
# scatter plots only include players that have non zero values for both metrics and 30 min TOI over 4 games -----------
zone_entry_plotting = player_entry_summ[player_entry_summ.\
    apply(lambda x: (x['clean_success_entry_poss_level_per_min']>0) & \
          (x['succ_entry_lt_.2_per_min']>0) & (x['toi_in_min']>30), axis=1)]

x_entry = copy.copy(zone_entry_plotting[['clean_success_entry_poss_level_per_min']]).to_numpy().reshape(-1)
y_entry = copy.copy(zone_entry_plotting[['succ_entry_lt_.2_per_min']]).to_numpy().reshape(-1)
team_entry = copy.copy(zone_entry_plotting[['team_name']]).to_numpy().reshape(-1)
player_entry = copy.copy(zone_entry_plotting[['Player']]).to_numpy().reshape(-1)
color_entry = np.where(team_entry == 'Women - Canada', "red", "blue")

fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(111)
plt.scatter(x_entry, y_entry, c = color_entry, s = 50)
plt.title("Carried Zone Entries Leading to Shots and/or\nOffensive Zone Possession at Even Strength", 
          fontsize = 35)
plt.xlabel("Valuable Zone Entries, Per Minute of TOI",fontsize = 20)
plt.ylabel("Valuable Difficult Zone Entries, Per Minute of TOI",fontsize = 20)
for i, txt in enumerate(player_entry):
    if txt == "Alex Carpenter":
        ax.text(x_entry[i], y_entry[i], txt, fontsize = 15, horizontalalignment = 'right',
                verticalalignment = 'bottom')
        continue;
    elif txt == "Abby Roque":
        ax.text(x_entry[i], y_entry[i], txt, fontsize = 15, horizontalalignment = 'right',
                verticalalignment = 'top')
        continue;
    elif txt == "Blayre Turnbull":
        ax.text(x_entry[i], y_entry[i], txt, fontsize = 15, horizontalalignment = 'right',
                verticalalignment = 'bottom')
        continue;
    elif txt == "Laura Stacey":
        ax.text(x_entry[i], y_entry[i], txt, fontsize = 15, horizontalalignment = 'left',
                verticalalignment = 'top')
        continue;  

    ax.text(x_entry[i], y_entry[i], txt, fontsize = 15, horizontalalignment = 'left',
            verticalalignment = 'bottom')
    
plt.savefig("charts/zone_entry_chart.png", format="png", bbox_inches='tight')
