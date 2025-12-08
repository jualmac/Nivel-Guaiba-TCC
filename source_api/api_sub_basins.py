import os
import json
import logging
import requests
import pandas as pd
from source_api.api_auth import get_auth
from db_handler import DBConnection
from util import configure_logging

logger = configure_logging(__name__)

# Get proper HidroWeb Token;
token = get_auth()

# Base URL;
url = "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/HidroSubBacia/v1"

# Headers with the authorization token;
headers = {
    "Authorization": f"Bearer {token}"
}

# Create request;
response = requests.get(url, headers=headers)

# Request Reponse;
if response.status_code == 200:
    stations = response.json()
    logger.info("Station data collected!")

    #Turn station json into a proper dataframe;
    df = pd.DataFrame(stations['items'])

    db_handler = DBConnection()
    db_handler.write(df, 'sub_basins', inplace=True)
else:
    logger.error("Request failed with status code: %s", response.status_code)
    logger.error("Response text: %s", response.text)
logger.info("DONE")