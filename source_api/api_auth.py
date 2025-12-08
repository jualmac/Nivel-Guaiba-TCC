import os
import json
import logging
import pandas as pd
from dotenv import load_dotenv
import requests
from util import configure_logging

logger = configure_logging(__name__)

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
    logger.info("Attempting connection...")
    response = requests.get(url, headers=headers)

    # Request Reponse:
    if response.status_code == 200:
        data = response.json()
        logger.info("Credentials acquired: %s", data)
        return(data['items']['tokenautenticacao'])
    else:
        logger.error("Request failed with status code %s", response.status_code)
        return 0