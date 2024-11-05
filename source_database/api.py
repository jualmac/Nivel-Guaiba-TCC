import os
import json
import pandas as pd
from dotenv import load_dotenv
import requests

load_dotenv()

TOKEN = os.getenv("TOKEN")
if not TOKEN:
    raise ValueError("TOKEN is not set in the environment variables")
else:
    print(TOKEN)

# Base URL;
url = "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/HidrosatInventarioEstacoes/v1"

# Headers with the authorization token;
headers = {
    "Authorization": f"Bearer {TOKEN}"
}

# Create request;
response = requests.get(url, headers=headers)

# Request Reponse;
if response.status_code == 200:
    print("Response data:", response.json())
else:
    print("Request failed with status code:", response.status_code)
    print("Response text:", response.text)