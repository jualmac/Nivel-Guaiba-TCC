import os
import json
import pandas as pd
from dotenv import load_dotenv
import requests

load_dotenv()

ID=os.getenv("ID")
PASS=os.getenv("PASS")

def get_auth() -> str:
    '''
    Gets the proper token necessary for the other API's from the HidroWeb service
    '''
    # Base URL;
    url = "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/OAUth/v1"

    headers = {
        'accept': '*/*',
        'Identificador': ID,
        'Senha': PASS
    }

    # Create request;
    print("[api_auth.py] Atempting connection...")
    response = requests.get(url, headers=headers)

    # Request Reponse:
    if response.status_code == 200:
        data = response.json()
        print(f"[api_auth.py] Credentials adquired! {data}")
        return(data['items']['tokenautenticacao'])
    else:
        print(f"[api_auth.py] Request failed with status code {response.status_code}")
        return 0