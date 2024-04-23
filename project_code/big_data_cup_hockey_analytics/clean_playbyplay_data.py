"""
Summary:
Clean play-by-play data, add fields about possession and zone entry details, and limit data to plays before zone entries. 
"""

%reset -sf
import pandas as pd
import numpy as np
import re
import os
import copy
import random

# set working directory, for use across all code files
os.chdir('C:/Users/chris/Documents/Data/big_data_cup')

# Data Cleaning and Feature Engineering ----------------------------------------------------------------------------------------

# play by play data
data = pd.read_csv(os.path.join(os.getcwd(), 'import/BDC_2024_Womens_Data.csv'))

# create new fields
build = data[['Clock', 'X Coordinate', 'Y Coordinate', 
              'X Coordinate 2', 'Y Coordinate 2', 'Team', 'Player','Player 2','Event', 'Detail 1', 
              'Home Team Skaters', 'Away Team Skaters', 'Home Team', 'Away Team']] .\
   rename(columns = {'X Coordinate':'X_Coordinate', 'Y Coordinate':'Y_Coordinate',
                      'X Coordinate 2':'X_Coordinate_2', 'Y Coordinate 2':'Y_Coordinate_2',
                      'Detail 1':'Detail_1', 'Player 2':'Player_2',
                      'Home Team Skaters':'ht_skaters', 'Away Team Skaters':'at_skaters',
                      'Home Team':'ht', 'Away Team':'at'}).\
 assign(
        clock_minutes = lambda x: x.Clock.str.extract(r'(\d{1,2})\:.+$').astype('int'),
        clock_seconds = lambda x: x.Clock.str.extract(r'.+\:(\d{2}$)').astype('int'),
        clock_in_sec = lambda x: (x.clock_minutes * 60) + x.clock_seconds,
        team_lag = lambda x: x.Team.shift(1),
         team_lead = lambda x: x.Team.shift(-1),
         event_lead = lambda x: x.Event.shift(-1),
         event_lag = lambda x: x.Event.shift(1),
         event_detail_lead = lambda x: x.Detail_1.shift(-1),
         clock_lead = lambda x: x.Clock.shift(-1),
         x_coord_lag = lambda x: x.X_Coordinate.shift(1))
build['active_team_skaters'] = np.select([build['ht']==build['Team']], 
[build['ht_skaters']], default=build['at_skaters']) 
build['non_active_team_skaters'] = np.select([build['ht']==build['Team']], 
[build['at_skaters']], default=build['ht_skaters']) 

# create field that indicates situation e.g. even strength, PP, PK
build['active_team_sich'] = np.select([(build['active_team_skaters']==5) & \
        (build['non_active_team_skaters']==5),
        (build['active_team_skaters']<5) & (build['active_team_skaters']==build['non_active_team_skaters']),
        (build['active_team_skaters']>build['non_active_team_skaters']),
        (build['active_team_skaters']<build['non_active_team_skaters'])
        ], 
        ['Even Strength', 'OT/4on4', 'PP', 'PK'
        ], default='Other') 

# get start time of next event
build['clock_lead'] = np.select([build['clock_lead'].isna(), (build['clock_lead'] == '20:00') & \
                                  (build['clock_in_sec'] < 1200)], 
['0:00', '0:00'], default=build['clock_lead'])

# convert clock time to seconds
build = build.\
    assign(
          clock_lead_minutes = lambda x: x.clock_lead.str.extract(r'(\d{1,2})\:.+$').astype('int'),
          clock_lead_seconds = lambda x: x.clock_lead.str.extract(r'.+\:(\d{2}$)').astype('int'),
          clock_lead_in_sec = lambda x: (x.clock_lead_minutes * 60) + x.clock_lead_seconds
        )
 


# create a variable that indicates when a new possession starts
conditions  = [ (build["Event"] == "Puck Recovery") & (build["team_lag"] != build["Team"]),
               build["Event"] == "Takeaway", 
               # faceoff indicates new possession
               build["Event"] == "Faceoff Win",
               # when leave offensive zone into neutral zone then count that as a new possession
               (build['Team'] == build['team_lag']) & (build['X_Coordinate']<125) & (build['x_coord_lag']>=125),
               
               # penalty is not new possession
               build["Event"] == "Penalty Taken",
               
               # puck recoveries that aren't changes in possession
               (build["Event"] == "Puck Recovery") & (build["team_lag"] == build["Team"]),
      
                   # if pass then possession stays same
                 ((build["Event"] == "Play") | (build["Event"] == "Incomplete Play")) & \
                   (build["team_lag"] == build["Team"]),
                   
                   # if incomplete pass but still recover then possession stays same
                   (build["Event"] == "Incomplete Play") & \
                     (build["team_lead"] == build["Team"]),
                   
                np.logical_or(build["Event"] == "Dump In/Out", build["Event"] == "Zone Entry") & \
                    (build["team_lag"] == build["Team"]),
                    np.logical_or(build["Event"] == "Shot", build["Event"] == "Goal") & \
                        (build["team_lag"] == build["Team"])]
choices     = [ 1,1,1,1,0,0, 0,0,0,0]
    
build["possess_change_flag"] = np.select(conditions, choices, default=1)


# create a unique id for each possession
build = build .\
    assign(unique_poss_id = lambda x: x.possess_change_flag.cumsum())
    

# create a variable (grp_order) that indicates the order of events in a possession
build= pd.concat([build, build.\
    groupby(['unique_poss_id'], as_index=False).cumcount()+1], axis=1) .\
    rename(columns = {0:"grp_order"})
    
 # flag at what time (grp order) the entry happened and also what type of entry
build['entry_order'] = np.select([build['Event']=='Zone Entry'], 
[build['grp_order']], default=np.nan)

build['entry_carry'] = np.select([(build['Event']=='Zone Entry') & (build['Detail_1']=='Carried')], 
[1], default=0)

build['entry_comp_pass'] = np.select([(build['Event']=='Zone Entry') & (build['Detail_1']=='Played') &\
                    (build['event_lag'] == 'Play')], 
[1], default=0)
    
build['entry_incomp_pass'] = np.select([(build['Event']=='Zone Entry') & (build['Detail_1']=='Played') &\
                    (build['event_lag'] == 'Incomplete Play')], 
[1], default=0)

build['entry_dump'] = np.select([(build['Event']=='Zone Entry') & (build['Detail_1']=='Dumped')], 
[1], default=0)
                     
build['shot_flag'] = np.select([(build['Event']=='Shot') | (build['Event']=='Goal')], 
[1], default=0)

build['shot_order'] = np.select([(build['Event']=='Shot') | (build['Event']=='Goal')], 
[build['grp_order']], default=np.nan)

build['entry_time'] = np.select([build['Event']=='Zone Entry'], 
[build['clock_in_sec']], default=np.nan)

build['penalty_flag'] = np.select([build['Event']=='Penalty Taken'], 
[1], default=0)

# make possession level flags, e.g. did a possession have a shot, a carried zone entry, pass zone entry, etc.
build = build.\
    assign(entry_order_poss_level=build.groupby(['unique_poss_id'], as_index=False)['entry_order'].transform('max'),
       poss_id_max_grp_order=build.groupby(['unique_poss_id'], as_index=False)['grp_order'].transform('max'),
       entry_carry_poss_level = build.groupby(['unique_poss_id'], as_index=False)['entry_carry'].transform('max'),
       entry_comp_pass_poss_level = build.groupby(['unique_poss_id'], as_index=False)['entry_comp_pass'].transform('max'),
       entry_incomp_pass_poss_level = build.groupby(['unique_poss_id'], as_index=False)['entry_incomp_pass'].transform('max'),
       entry_dump_poss_level = build.groupby(['unique_poss_id'], as_index=False)['entry_dump'].transform('max'),
       
       shot_flag_poss_level = build.groupby(['unique_poss_id'], as_index=False)['shot_flag'].transform('max'),
       shot_order_poss_level=build.groupby(['unique_poss_id'], as_index=False)['shot_order'].transform('min'),
       entry_time_poss_level = build.groupby(['unique_poss_id'], as_index=False)['entry_time'].transform('max'),
       penalty_flag_poss_level = build.groupby(['unique_poss_id'], as_index=False)['penalty_flag'].transform('max')
       )
    
# filter out possessions with penalties because don't want that to count as zone entry attempt
build = build[build.\
    apply(lambda x: (x['penalty_flag_poss_level'] ==0), axis=1)]

# get end time of possession
build['poss_end_time'] = np.select([(build['Event'].isin(['Incomplete Play', 'Dump In/Out', 'Faceoff Win', 'Shot'])) & \
                (build['poss_id_max_grp_order']==build['grp_order']), build['poss_id_max_grp_order']==build['grp_order']], 
[build['clock_in_sec'], build['clock_lead_in_sec']], default=np.nan)

build = build.\
    assign(poss_end_time_poss_level = build.groupby(['unique_poss_id'], as_index=False)['poss_end_time'].transform('min'))
    
# flag a possession if it had a zone entry that led to a shot attempt or at least 5 seconds of offensive zone possession
build['clean_success_entry_poss_level'] = np.select([
    ((build['entry_comp_pass_poss_level'] == 1) | (build['entry_carry_poss_level'] == 1) |\
     (build['entry_incomp_pass_poss_level'] == 1) | (build['entry_dump_poss_level'] == 1)
     ) & \
    ((build['poss_end_time_poss_level'] <= build['entry_time_poss_level']-5) | (build['shot_flag_poss_level'] == 1))],
    [1], default=0)

# output a datset that contains all events including zone entry plays 
with_all_zone_entries = copy.copy(build)

with_all_zone_entries.to_csv(os.path.join(os.getcwd(), 'datasets/playbyplay_data_with_zone_entries.csv'))


# filter out zone entries from dataset before making stick handle start and end points
# but keep carried zone entries because they are 
# informative for how the stick handle progresses geographically 
build = build[build.\
      apply(lambda x: (x['Event'] != 'Zone Entry') | (x['Detail_1'] == 'Carried'), axis=1)]
    
# make variables to show the start coordinate of the next play
build = build.\
    assign(
x1coord_lead = lambda x: x.X_Coordinate.shift(-1),
y1coord_lead = lambda x: x.Y_Coordinate.shift(-1)
)

# make variables to represent the starting coordinate of a zone entry
# if puck recovery or takeaway or carried zone entry then the location of the event is also the starting location of the stick handle that follows
# if incomplete pass, shot, etc, then there is no stick handle that follows
# if last event is a pass but then other team recovers the puck assume the first team just left the puck there so no stick handling followed the pass
# if there is a pass to someone then the receiving location of the pass is also the starting location of the stick handle that follows
conditions3  = [ build['Event'].isin(['Puck Recovery', 'Takeaway', 'Zone Entry']), 
                
                build['Event'].isin(['Incomplete Play',
  'Faceoff Win', 'Dump In/Out', 'Shot']), (build['Event']=='Play') & (build['event_lead']=='Puck Recovery') &\
                (build['team_lead']!=build['Team']), build["X_Coordinate_2"].notna()]
choicesxstart     = [build["X_Coordinate"],np.nan,np.nan, build["X_Coordinate_2"]]
choicesystart    = [build["Y_Coordinate"],np.nan,np.nan, build["Y_Coordinate_2"]]
    
build["stick_handle_start_x"] = np.select(conditions3, choicesxstart, default = np.nan)
build["stick_handle_start_y"] = np.select(conditions3, choicesystart, default = np.nan)

# make variables to represent the ending coordinate of a zone entry as the 
# starting location of the next event
# account for cases when possession changes right after the possession
conditions2  = [ build['Event'].isin(['Incomplete Play', 'Dump In/Out', 'Faceoff Win', 'Shot']),
    (build["team_lead"] != build["Team"]) &  (build['event_lead'].isin(['Takeaway','Puck Recovery'])),
    build["team_lead"] != build["Team"]
    ]
choicesxend     = [np.nan, 200 - build["x1coord_lead"], np.nan]
choicesyend    = [np.nan, 85 - build["y1coord_lead"], np.nan]
    
build["stick_handle_end_x"] = np.select(conditions2, choicesxend, default = build["x1coord_lead"])
build["stick_handle_end_y"] = np.select(conditions2, choicesyend, default = build["y1coord_lead"])

build = build.\
    drop(columns = ["team_lag",
    "team_lead",
    "event_lead",
    "event_lag",
    "event_detail_lead",
    "x1coord_lead",
    "y1coord_lead"])


# get a dataset with all stick handles
stick_handle = build[['stick_handle_start_x', 'stick_handle_start_y','stick_handle_end_x', 'stick_handle_end_y',
                      'grp_order', 'unique_poss_id', 'Team', 'Player', 'Player_2', 'active_team_sich']] .\
    assign(grp_order_stick_handle = lambda x: x.grp_order + 0.5,
           Event = 'Stick Handle') .\
    drop(columns=['grp_order']).\
    rename(columns = {'grp_order_stick_handle':'grp_order'}).\
    dropna(subset=['stick_handle_start_x', 'stick_handle_end_x']) .\
    rename(columns = {'stick_handle_start_x':'X_Coordinate',
                      'stick_handle_start_y':'Y_Coordinate',
                      'stick_handle_end_x':'X_Coordinate_2',
                      'stick_handle_end_y':'Y_Coordinate_2'})

# define the name of the stick handler
stick_handle['player_stick_handle'] = np.select([stick_handle['Player_2'].isna()], 
[stick_handle['Player']], default=stick_handle['Player_2'])

stick_handle = stick_handle.\
    drop(columns=['Player', 'Player_2']).\
    rename(columns = {'player_stick_handle':'Player'})

# combine stick handle dataset with dataset of all other events
full_build = pd.concat([build, stick_handle], axis = 0) .\
    sort_values(by=['unique_poss_id', 'grp_order']).\
    assign(clock_lag = lambda x: x.Clock.shift(1),
           clock_lead = lambda x: x.Clock.shift(-1)).\
        reset_index(drop = True)

 # now that stick handles are added to dataset, do a second pass at possession level 
 # flags for when a zone entry happened, which overwrites the first flag, to fill in values for stick handle rows
full_build = full_build.\
    assign(entry_order_poss_level=full_build.groupby(['unique_poss_id'], as_index=False)['entry_order_poss_level'].transform('max'),
           poss_id_max_grp_order=full_build.groupby(['unique_poss_id'], as_index=False)['poss_id_max_grp_order'].transform('max'),
           poss_id_w_sh_max_grp_order=full_build.groupby(['unique_poss_id'], as_index=False)['grp_order'].transform('max'),
           entry_carry_poss_level = full_build.groupby(['unique_poss_id'], as_index=False)['entry_carry_poss_level'].transform('max'),
           entry_comp_pass_poss_level = full_build.groupby(['unique_poss_id'], as_index=False)['entry_comp_pass_poss_level'].transform('max'),
           entry_incomp_pass_poss_level = full_build.groupby(['unique_poss_id'], as_index=False)['entry_incomp_pass_poss_level'].transform('max'),
           entry_dump_poss_level = full_build.groupby(['unique_poss_id'], as_index=False)['entry_dump_poss_level'].transform('max'),
           shot_flag_poss_level = full_build.groupby(['unique_poss_id'], as_index=False)['shot_flag_poss_level'].transform('max'),
           clean_success_entry_poss_level = full_build.groupby(['unique_poss_id'], as_index=False)['clean_success_entry_poss_level'].transform('max')
           )


full_build['x_coord_first_poss_event'] = np.select([full_build['grp_order']==1], 
[full_build['X_Coordinate']], default=np.nan)

full_build = full_build.\
    assign(x_coord_first_poss_event_poss_level = full_build.groupby(['unique_poss_id'], as_index=False)['x_coord_first_poss_event'].\
           transform('min'),
           event_lead = lambda x: x.Event.shift(-1),
           team_lead = lambda x: x.Team.shift(-1))

# clean the coordinates of events
full_build['X_Coordinate_2'] = np.select([full_build['X_Coordinate_2'].isna()], 
[full_build['X_Coordinate']], default=full_build['X_Coordinate_2'])

full_build['Y_Coordinate_2'] = np.select([full_build['Y_Coordinate_2'].isna()], 
[full_build['Y_Coordinate']], default=full_build['Y_Coordinate_2'])

# add features for use in prediction model
full_build = full_build.\
    assign(vert_move = lambda x: x.X_Coordinate_2-x.X_Coordinate,
           lat_move = lambda x: abs(x.Y_Coordinate_2-x.Y_Coordinate),
           distance = lambda x: (x.vert_move**2 + x.lat_move**2)**(1/2))

# filter out stick handles where there was no coordinate movement
# and also filter out stick handles that were less than 5 feet in distance
full_build = full_build[full_build.apply(\
    lambda x: (x['Event'] != 'Stick Handle') |\
        (x['X_Coordinate'] != x['X_Coordinate_2']) | (x['Y_Coordinate'] != x['Y_Coordinate_2']) , axis=1)] 
    
full_build = full_build[full_build.apply(\
    lambda x: (x.distance >= 5) | (x.Event != "Stick Handle"), axis=1)] 
        
full_build.to_csv(os.path.join(os.getcwd(), 'datasets/playbyplay_data_with_carried_entries.csv'))

# filter out events that happen after the zone entry, to get a dataset containing just defensive/neutral zone possessions
full_pre_entry_build = copy.copy(full_build[full_build.\
    apply(lambda x: ((x['entry_carry_poss_level'] == 1) & (x['grp_order'] <= x['entry_order_poss_level']-1)) | \
        ((x['entry_dump_poss_level'] == 1) & (x['grp_order'] <= x['entry_order_poss_level']-1.5)) | \
        ((x['entry_comp_pass_poss_level'] == 1) & (x['grp_order'] <= x['entry_order_poss_level']-1.5)) | \
            ((x['entry_incomp_pass_poss_level'] == 1) & (x['grp_order'] <= x['entry_order_poss_level']-1.5)) | \
        ((pd.isna(x['entry_order_poss_level'])) & (x['grp_order'] < x['poss_id_w_sh_max_grp_order'])) | \
            # if there is a takeaway then keep the last stick handle before takeaway as well
   ((pd.isna(x['entry_order_poss_level'])) & (x['grp_order'] == x['poss_id_w_sh_max_grp_order']) &\
    (x['event_lead'] == "Takeaway") & (x['Team'] != x['team_lead']) & (x['Event'] == 'Stick Handle'))
        , axis=1)])

# filter out possessions that started in offensive zone 
# and any parts of possession that could be in offesnsive zone without zone entry
full_pre_entry_build = full_pre_entry_build[full_pre_entry_build.\
    apply(lambda x: (x['x_coord_first_poss_event_poss_level'] < 125) &\
          (x['X_Coordinate'] < 125), axis=1)]
    
 # filter out weird possessions where zone entries not properly accounted for in data with
full_pre_entry_build = full_pre_entry_build.\
assign(max_x_coord_1 = full_pre_entry_build.groupby(['unique_poss_id'], as_index=False)['X_Coordinate'].transform('max'),
       max_x_coord_2 = full_pre_entry_build.groupby(['unique_poss_id'], as_index=False)['X_Coordinate_2'].transform('max'))
full_pre_entry_build = full_pre_entry_build[full_pre_entry_build.\
    apply(lambda x: (x['max_x_coord_1']<=125 ) & (x['max_x_coord_2']<=125 ), axis=1)]

full_pre_entry_build = full_pre_entry_build.\
    reset_index(drop = True)

full_pre_entry_build.to_csv(os.path.join(os.getcwd(), 'datasets/playbyplay_data_pre_zone_entry.csv'))

    
