"""
This is util package for storing constants, dictionaries and util functions;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import logging
import os
import numpy as np
import pandas as pd
import subprocess
import shutil
from datetime import datetime, timedelta
from typing import Callable, Iterator

########################################################################################################################
#                                                                  
# CONSTANTS
#
########################################################################################################################
# START_DATE = '2018-08-01'
START_DATE = '2026-01-01'
END_DATE = '2026-02-01'
BASE_URL = "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/"

########################################################################################################################
#                                                                  
# FUNCTIONS
#
########################################################################################################################
logger = logging.getLogger(__name__)


def configure_logging(logger_name: str | None = None, log_file: str = "logs/pipeline.log", level: int = logging.INFO):
    """
    Configure application logging with both console and file handlers.

    Uses root logger to avoid duplicate handler setup across modules. Safe to call multiple
    times; handlers are only attached once.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        root_logger.setLevel(level)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    return logging.getLogger(logger_name or __name__)


def is_valid(number):
    if (
        number is None
        or np.isnan(number)
        or number == "None"
        or number == "NaN"
        or number == "nan"
    ):
        return False
    else:
        return True


def flatten(list):
    flatten_list = []
    for sublist in list:
        for item in sublist:
            flatten_list.append(item)
    return flatten_list


def convert_to_float(value: str) -> float:
    """
    Cleans a string value by removing commas and converts it to float if possible. If the input value is not a string 
    containing commas, it is returned unchanged;

    Parameters:
        - value (str): Input value (string or numeric);
    
    Returns:
        - float: Converted float value if applicable, otherwise returns the input value unchanged;
    """
    
    if value is None:
        return None
    elif isinstance(value, str) and ',' in value:
        cleaned_value = value.replace(',', '')
        try:
            return float(cleaned_value)
        except ValueError:
            return value
    else:
        return float(value)


def get_device_config(mode: str, model_type: str) -> dict:
    """
    Converts mode parameter to appropriate device settings for XGBoost and LightGBM,
    validating against actual hardware availability if possible.
    
    Parameters:
        mode: Training mode - 'CPU', 'GPU', or 'CUDA';
        model_type: Type of model - 'xgboost' or 'lightgbm';
    
    Returns:
        dict: Device configuration for the model;
    """
    mode = mode.upper()
    model_type = model_type.lower()
    
    # Check availability if GPU requested
    if mode in ['GPU', 'CUDA']:
        if not is_gpu_available():
            logger.warning(f"{mode} mode requested but no GPU detected via nvidia-smi. Falling back to CPU.")
            mode = 'CPU'

    if mode == 'CPU':
        if model_type == 'xgboost':
            return {'device': 'cpu', 'tree_method': 'hist'}
        elif model_type == 'lightgbm':
            return {'device': 'cpu'}
        else:
            # Fallback for other models or raise error
             return {'device': 'cpu'}
    
    elif mode in ['GPU', 'CUDA']:
        if model_type == 'xgboost':
            # Modern XGBoost prefers device='cuda'
            return {'device': 'cuda', 'tree_method': 'hist'}
        elif model_type == 'lightgbm':
            # LightGBM usually expects device='gpu'
            return {'device': 'gpu'}
        else:
             raise ValueError(f"Invalid model_type: {model_type}. Choose from 'xgboost' or 'lightgbm'")
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose from 'CPU', 'GPU', or 'CUDA'")


def is_cpu_mode(mode: str) -> bool:
    """
    Check if the mode is CPU-only. Validates actual hardware if mode is GPU/CUDA.
    
    Parameters:
        mode: Training mode - 'CPU', 'GPU', or 'CUDA';
    
    Returns:
        bool: True if CPU mode or if GPU requested but unavailable; False otherwise.
    """
    mode = mode.upper()
    if mode == 'CPU':
        return True
    
    if mode in ['GPU', 'CUDA']:
        return not is_gpu_available()
    return True


def is_gpu_available() -> bool:
    """
    Checks if a GPU is available on the system by looking for nvidia-smi.
    
    Returns:
        bool: True if nvidia-smi is found and runs successfully, False otherwise.
    """
    if shutil.which('nvidia-smi') is None:
        return False
    try:
        subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


########################################################################################################################
#                                                                  
# DICTIONARIES
#
########################################################################################################################
STATION_COLS = [
    'Data_Hora_Medicao',
    'codigoestacao',
    'Cota_Adotada',
    'Cota_Adotada_Status',
    'Chuva_Adotada',
    'Chuva_Adotada_Status',
    'Temperatura_Interna',
    'Temperatura_Interna_Status',
    'Vazao_Adotada',
    'Vazao_Adotada_Status',
    'Altitude',
    'Area_Drenagem',
    'Latitude',
    'Longitude',
    'Rio_Codigo',
    ]

AGG_DICT = {'Cota_Adotada': 'mean',
            'Cota_Adotada_Status': 'max',  
            'Chuva_Adotada': 'mean',
            'Chuva_Adotada_Status': 'max',  
            'Vazao_Adotada': 'mean',
            'Vazao_Adotada_Status': 'max',  
            'Temperatura_Interna': 'mean',
            'Temperatura_Interna_Status': 'max',
            'Altitude': 'first',
            'Area_Drenagem': 'first',
            'Latitude': 'first',
            'Longitude': 'first',
            'Rio_Codigo': 'first',
            }

STATION_CODES = {
    'Guaíba (CAIS MAUÁ C6 + USINA DO GASÔMETRO)': '87450004',
    'Jacuí (RIO PARDO)': '85900000',
    'Gravataí (PASSO DAS CANOAS - AUXILIAR)': '87399000',
    'Taquari (MUÇUM)': '86510000',
    'Taquari (ENCANTADO)': '86720000',
    'Sinos (SÃO LEOPOLDO)': '87382000',
    'Sinos (CAMPO BOM)': '87380000',
    'Caí (LINHA GONZAGA)': '87150000',
    'Caí (BARCA DO CAÍ)': '87170000',
}

BRAZILIAN_STATES = {
    'AC': 'Acre',
    'AL': 'Alagoas',
    'AP': 'Amapá',
    'AM': 'Amazonas',
    'BA': 'Bahia',
    'CE': 'Ceará',
    'DF': 'Distrito Federal',
    'ES': 'Espírito Santo',
    'GO': 'Goiás',
    'MA': 'Maranhão',
    'MT': 'Mato Grosso',
    'MS': 'Mato Grosso do Sul',
    'MG': 'Minas Gerais',
    'PA': 'Pará',
    'PB': 'Paraíba',
    'PR': 'Paraná',
    'PE': 'Pernambuco',
    'PI': 'Piauí',
    'RJ': 'Rio de Janeiro',
    'RN': 'Rio Grande do Norte',
    'RS': 'Rio Grande do Sul',
    'RO': 'Rondônia',
    'RR': 'Roraima',
    'SC': 'Santa Catarina',
    'SP': 'São Paulo',
    'SE': 'Sergipe',
    'TO': 'Tocantins',
    }

COUNTRY_NAMES_PT = {
    "AF": "Afeganistão",
    "AL": "Albânia",
    "DZ": "Argélia",
    "AS": "Samoa Americana",
    "AD": "Andorra",
    "AO": "Angola",
    "AG": "Antígua e Barbuda",
    "AR": "Argentina",
    "AM": "Armênia",
    "AU": "Austrália",
    "AT": "Áustria",
    "AZ": "Azerbaijão",
    "BS": "Bahamas",
    "BH": "Bahrain",
    "BD": "Bangladesh",
    "BB": "Barbados",
    "BY": "Bielorrússia",
    "BE": "Bélgica",
    "BZ": "Belize",
    "BJ": "Benin",
    "BT": "Butão",
    "BO": "Bolívia",
    "BA": "Bósnia e Herzegovina",
    "BW": "Botsuana",
    "BR": "Brasil",
    "BN": "Brunéi",
    "BG": "Bulgária",
    "BF": "Burkina Faso",
    "BI": "Burundi",
    "CV": "Cabo Verde",
    "KH": "Camboja",
    "CM": "Camarões",
    "CA": "Canadá",
    "KY": "Ilhas Cayman",
    "CF": "República Centro-Africana",
    "TD": "Chade",
    "CL": "Chile",
    "CN": "China",
    "CO": "Colômbia",
    "KM": "Comores",
    "CG": "Congo",
    "CD": "República Democrática do Congo",
    "CR": "Costa Rica",
    "HR": "Croácia",
    "CU": "Cuba",
    "CY": "Chipre",
    "CZ": "República Tcheca",
    "DK": "Dinamarca",
    "DJ": "Djibuti",
    "DM": "Dominica",
    "DO": "República Dominicana",
    "EC": "Equador",
    "EG": "Egito",
    "SV": "El Salvador",
    "GQ": "Guiné Equatorial",
    "ER": "Eritreia",
    "EE": "Estônia",
    "SZ": "Eswatini",
    "ET": "Etiópia",
    "FJ": "Fiji",
    "FI": "Finlândia",
    "FR": "França",
    "GA": "Gabão",
    "GM": "Gâmbia",
    "GE": "Geórgia",
    "DE": "Alemanha",
    "GH": "Gana",
    "GR": "Grécia",
    "GD": "Granada",
    "GT": "Guatemala",
    "GN": "Guiné",
    "GW": "Guiné-Bissau",
    "GY": "Guiana",
    "HT": "Haiti",
    "HN": "Honduras",
    "HK": "Hong Kong",
    "HU": "Hungria",
    "IS": "Islândia",
    "IN": "Índia",
    "ID": "Indonésia",
    "IR": "Irã",
    "IQ": "Iraque",
    "IE": "Irlanda",
    "IL": "Israel",
    "IT": "Itália",
    "JM": "Jamaica",
    "JP": "Japão",
    "JE": "Jersey",
    "JO": "Jordânia",
    "KZ": "Cazaquistão",
    "KE": "Quênia",
    "KI": "Quiribati",
    "KP": "Coreia do Norte",
    "KR": "Coreia do Sul",
    "KW": "Kuwait",
    "KG": "Quirguistão",
    "LA": "Laos",
    "LV": "Letônia",
    "LB": "Líbano",
    "LS": "Lesoto",
    "LR": "Libéria",
    "LY": "Líbia",
    "LI": "Liechtenstein",
    "LT": "Lituânia",
    "LU": "Luxemburgo",
    "MO": "Macau",
    "MG": "Madagascar",
    "MW": "Malawi",
    "MY": "Malásia",
    "MV": "Maldivas",
    "ML": "Mali",
    "MT": "Malta",
    "MH": "Ilhas Marshall",
    "MQ": "Martinica",
    "MR": "Mauritânia",
    "MU": "Maurício",
    "YT": "Mayotte",
    "MX": "México",
    "FM": "Micronésia",
    "MD": "Moldávia",
    "MC": "Mônaco",
    "MN": "Mongólia",
    "ME": "Montenegro",
    "MA": "Marrocos",
    "MZ": "Moçambique",
    "MM": "Mianmar",
    "NA": "Namíbia",
    "NR": "Nauru",
    "NP": "Nepal",
    "NL": "Países Baixos",
    "NZ": "Nova Zelândia",
    "NI": "Nicarágua",
    "NE": "Níger",
    "NG": "Nigéria",
    "NO": "Noruega",
    "OM": "Omã",
    "PK": "Paquistão",
    "PW": "Palau",
    "PS": "Palestina",
    "PA": "Panamá",
    "PG": "Papua-Nova Guiné",
    "PY": "Paraguai",
    "PE": "Peru",
    "PH": "Filipinas",
    "PL": "Polônia",
    "PT": "Portugal",
    "PR": "Porto Rico",
    "QA": "Catar",
    "RO": "Romênia",
    "RU": "Rússia",
    "RW": "Ruanda",
    "WS": "Samoa",
    "SM": "São Marino",
    "ST": "São Tomé e Príncipe",
    "SA": "Arábia Saudita",
    "SN": "Senegal",
    "RS": "Sérvia",
    "SC": "Seicheles",
    "SL": "Serra Leoa",
    "SG": "Cingapura",
    "SK": "Eslováquia",
    "SI": "Eslovênia",
    "SB": "Ilhas Salomão",
    "SO": "Somália",
    "ZA": "África do Sul",
    "SS": "Sudão do Sul",
    "ES": "Espanha",
    "LK": "Sri Lanka",
    "SD": "Sudão",
    "SR": "Suriname",
    "SZ": "Suazilândia",
    "SE": "Suécia",
    "CH": "Suíça",
    "SY": "Síria",
    "TJ": "Tajiquistão",
    "TZ": "Tanzânia",
    "TH": "Tailândia",
    "TG": "Togo",
    "TK": "Tokelau",
    "TO": "Tonga",
    "TT": "Trinidad e Tobago",
    "TN": "Tunísia",
    "TR": "Turquia",
    "TM": "Turcomenistão",
    "TV": "Tuvalu",
    "UG": "Uganda",
    "UA": "Ucrânia",
    "AE": "Emirados Árabes Unidos",
    "GB": "Reino Unido",
    "US": "Estados Unidos",
    "UY": "Uruguai",
    "UZ": "Uzbequistão",
    "VU": "Vanuatu",
    "VE": "Venezuela",
    "VN": "Vietnã",
    "WF": "Wallis e Futuna",
    "EH": "Sahara Ocidental",
    "YE": "Iémen",
    "ZM": "Zâmbia",
    "ZW": "Zimbábue"
    }