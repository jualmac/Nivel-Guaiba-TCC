"""
"""
########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import os
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from util import convert_to_float, STATION_COLS, AGG_DICT, START_DATE, END_DATE, STATION_CODES

########################################################################################################################
#                                                                  
# FUNCTION
#
########################################################################################################################
def make_station_comparison(df: pd.DataFrame == None):
    # Rename;
    df_cpy = df.copy()
    
    if "Data_Hora_Medicao" in df_cpy.columns:
        df_cpy.set_index("Data_Hora_Medicao", inplace=True)

    # Create 3x3 subplot grid;
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=list(STATION_CODES.keys()),
        shared_xaxes=True,
        vertical_spacing=0.1
    )

    # Add trace for each station from main dataframe;
    row, col = 1, 1
    for station_name, station_code in STATION_CODES.items():
        station_df = df_cpy[df_cpy['codigoestacao'] == station_code]
        if 'Cota_Adotada' in station_df.columns and len(station_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=station_df.index,
                    y=station_df['Cota_Adotada'],
                    mode="lines",
                    name=station_name,
                    line=dict(width=1.5),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Move to next position in grid;
        col += 1
        if col > 3:
            col = 1
            row += 1

    # Update layout;
    fig.update_layout(
        height=900,
        title_text="Cota_Adotada - Comparação entre Estações",
        hovermode='x unified',
        template="plotly_white"
    )

    # Update y-axes labels;
    for i in range(1, 10):
        fig.update_yaxes(title_text="Cota_Adotada", row=(i-1)//3 + 1, col=(i-1)%3 + 1)

    # Update x-axes labels for bottom row;
    for i in range(7, 10):
        fig.update_xaxes(title_text="Data_Hora_Medicao", row=3, col=i-6)
    fig.show()