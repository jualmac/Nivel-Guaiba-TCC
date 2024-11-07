import os
import json
import requests
import pandas as pd
from source_database.auth import get_auth
from source_database.db_handler import connect_to_database, write_dataframe, close_connection

# Get proper HidroWeb Token;
token = get_auth()

# Base URL;
url = "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/HidrosatInventarioEstacoes/v1"

# Headers with the authorization token;
headers = {
    "Authorization": f"Bearer {token}"
}

# Create request;
response = requests.get(url, headers=headers)

# Request Reponse;
if response.status_code == 200:
    stations = response.json()
    with open('./data/stations.json', 'w') as file:
        json.dump(stations, file, indent=4)
    print('Station data colected!')

    #Turn station json into a proper dataframe;
    df = pd.DataFrame(stations['items'])

    db_connection = connect_to_database()
    dataframe = write_dataframe(df, 'stations')
    close_connection(db_connection)
else:
    print("Request failed with status code:", response.status_code)
    print("Response text:", response.text)
print('DONE')