import streamlit as st
from PIL import Image
import streamlit as st
import base64
import requests
import textwrap
st.set_page_config(page_title="Premierleague", page_icon="⚽")



st.title("⚽Premierleague Dashboard")

import requests
import pandas as pd

url = 'https://fantasy.premierleague.com/api/bootstrap-static/'

r = requests.get(url)
json = r.json()
json.keys()

fpl_elements_df = pd.DataFrame(json['elements'])
fpl_elements_types_df = pd.DataFrame(json['element_types'])
fpl_teams_df = pd.DataFrame(json['teams'])

fpl_slim_elements_df = fpl_elements_df[['web_name','team',"goals_scored"
                                        ,'element_type','now_cost','minutes','total_points','assists', 'clean_sheets', 'goals_conceded', 
                                        'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 'red_cards', 'saves', 'influence', 'creativity', 'threat', 'influence_rank','influence_rank_type',
                                        'creativity_rank','creativity_rank_type', 'threat_rank', 'threat_rank_type', 'corners_and_indirect_freekicks_order',
                                        'direct_freekicks_order','penalties_order']]
                                        
fpl_slim_elements_df['position'] = fpl_slim_elements_df.element_type.map(fpl_elements_types_df.set_index('id').singular_name)

fpl_slim_elements_df['team'] = fpl_slim_elements_df.team.map(fpl_teams_df.set_index('id').name)
    
fpl_slim_elements_df = fpl_slim_elements_df[['web_name','team',"goals_scored"
                                        ,'element_type','now_cost','minutes','total_points','assists', 'clean_sheets', 'goals_conceded', 
                                        'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 'red_cards', 'saves', 'influence', 'creativity', 'threat', 'influence_rank','influence_rank_type',
                                        'creativity_rank','creativity_rank_type', 'threat_rank', 'threat_rank_type', 'corners_and_indirect_freekicks_order','direct_freekicks_order', 'penalties_order']]
                                             
fpl_slim_elements_df['total_points'] = fpl_slim_elements_df.total_points.astype(
    float)
    
fpl_slim_elements_df.sort_values('total_points',ascending=False)


fpl_slim_elements_df = fpl_slim_elements_df.fillna('0')

fpl_slim_elements_df[['total_points','corners_and_indirect_freekicks_order','direct_freekicks_order', 'penalties_order']] = fpl_slim_elements_df[['total_points','corners_and_indirect_freekicks_order','direct_freekicks_order', 'penalties_order']].astype(int)

fpl_slim_elements_df['attack_point'] = fpl_slim_elements_df['goals_scored'] + fpl_slim_elements_df['assists']
fpl_slim_elements_df['rank'] = fpl_slim_elements_df['attack_point'].rank(method='min', ascending=False)
fpl_slim_elements_df['rank'] = fpl_slim_elements_df['rank'].astype(int)

st.dataframe(fpl_slim_elements_df)

st.header("⚽Soccoer Players")
players_select = st.selectbox("Select Player",fpl_slim_elements_df['web_name'])
players_team = fpl_slim_elements_df[['web_name','team']]
players_team = players_team.set_index(['web_name'])

st.header(players_select)

st.subheader(players_team.loc[players_select,"team"])


col1, col2, col3, col4= st.columns(4)


col1.metric("Rank", fpl_slim_elements_df.loc[fpl_slim_elements_df["web_name"]==players_select,"rank"])
col2.metric("Goals", fpl_slim_elements_df.loc[fpl_slim_elements_df["web_name"]==players_select,"goals_scored"])
col3.metric("Assists", fpl_slim_elements_df.loc[fpl_slim_elements_df["web_name"]==players_select,"assists"])
col4.metric("Attack Point", fpl_slim_elements_df.loc[fpl_slim_elements_df["web_name"]==players_select,"attack_point"])
st.markdown("---")
st.subheader('🎬Player Highlight')
if players_select == players_select:
    st.video('https://youtu.be/yJVYTOqjavc')
