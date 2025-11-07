"""
This is util package for storing constants, dictionaries and util functions;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import numpy as np
import pandas as pd
import random

########################################################################################################################
#                                                                  
# FUNCTIONS
#
########################################################################################################################
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

########################################################################################################################
#                                                                  
# CONSTANTS
#
########################################################################################################################
START_DATE = '2018-08-01'
END_DATE = '2025-10-05'

########################################################################################################################
#                                                                  
# DICTIONARIES
#
########################################################################################################################
STATIONS_COLS = {
    'Data_Hora_Medicao': 'date', 
    'codigoestacao': 'station_id', 
    'Cota_Adotada': 'level',
    'Cota_Adotada_Status': 'level_status',
    'Chuva_Adotada': 'rainfall',
    'Chuva_Adotada_Status': 'rainfall_status',
    'Chuva_Acumulada': 'rainfall_acc',
    'Chuva_Acumulada_Status': 'rainfall_acc_status',
    'Temperatura_Interna': 'temperature',
    'Temperatura_Interna_Status': 'temperature_status',
    'Vazao_Adotada': 'flow',
    'Vazao_Adotada_Status': 'flow_status',
    }

AGG_DICT = {'level': 'mean',
            'level_status': 'first',
            'rainfall': 'mean',
            'rainfall_status': 'first',
            'rainfall_acc': 'mean',
            'rainfall_acc_status': 'first',
            # 'flow': 'mean',
            # 'flow_status': 'first',
            'temperature': 'mean',
            'temperature_status': 'first',
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