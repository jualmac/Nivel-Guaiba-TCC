import os
import json
import requests
import pandas as pd
from source_database.auth import get_auth

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
    print("Response data:", response.json())
else:
    print("Request failed with status code:", response.status_code)
    print("Response text:", response.text)

print('DONE')