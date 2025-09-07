import os
import json
import requests
import pandas as pd
from source_database.api_auth import get_auth
from source_database.db_handler import DBConnection

# Get proper HidroWeb Token;
token = get_auth()

# Base URL;
url = "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/HidroRio/v1"

# Headers with the authorization token;
headers = {
    "Authorization": f"Bearer {token}"
}

# Create request;
response = requests.get(url, headers=headers)

# Request Reponse;
if response.status_code == 200:
    stations = response.json()
    print('Station data colected!')

    #Turn station json into a proper dataframe;
    df = pd.DataFrame(stations['items'])

    db_handler = DBConnection()
    db_handler.write(df, 'rivers', inplace=True)
else:
    print("Request failed with status code:", response.status_code)
    print("Response text:", response.text)
print('DONE')