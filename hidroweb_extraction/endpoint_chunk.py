"""
Shared helpers for endpoint scripts;
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
import argparse
import os
import logging
import requests
import numpy as np
import pandas as pd
from typing import Any
from datetime import datetime, timedelta
from typing import Callable, Iterator

from auth import get_auth
from endpoint_request import request_with_auth_retry
from db_make import ensure_duckdb_exists
from db_handler import DBConnection
from util import START_DATE, END_DATE, BASE_URL, configure_logging

logger = configure_logging(__name__)

########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
class Chunk_Hidroweb_API():
    def __init__(self) -> None:
        self.token = get_auth()
        self.start_date = START_DATE
        self.end_date = END_DATE
        self.inplace = True
        self.logger = logger
        self.date_filter_type = "DATA_LEITURA"
        self.station_param_name = "Código da Estação"
        self.date_filter_param_name = "Tipo Filtro Data"
        self.search_date_param_name = "Data de Busca (yyyy-MM-dd)"
        self.range_param_name = "Range Intervalo de busca"

    def iter_hidroweb_date_chunks(
        self,
        start_date: str | datetime, 
        end_date: str | datetime
        ) -> Iterator[dict]:
        """
        Split a date window into API-compatible chunks while respecting the 30-day hard limit.

        The HidroWeb endpoints accept only specific range labels, so this helper always chooses
        the largest valid chunk for the remaining window to minimize request count;
        """
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_dt = start_date

        if isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end_dt = end_date

        if start_dt > end_dt:
            raise ValueError("start_date cannot be after end_date")

        # Ordered from largest to smallest to always minimize number of requests;
        chunk_options = [
            ("DIAS_30", 30),
            ("DIAS_21", 21),
            ("DIAS_14", 14),
            ("DIAS_7", 7),
            ("DIAS_2", 2),
        ]

        current_date = start_dt
        while current_date <= end_dt:
            remaining_days = (end_dt - current_date).days + 1

            if remaining_days == 1:
                # API has no DIAS_1 option, so DIAS_2 must be used with one-day effective span;
                range_days = "DIAS_2"
                requested_chunk_days = 1
            else:
                range_days = "DIAS_2"
                requested_chunk_days = 2
                for label, days in chunk_options:
                    if remaining_days >= days:
                        range_days = label
                        requested_chunk_days = days
                        break

            # The final chunk may be shorter than requested if we are at the end of the interval;
            chunk_end_date = min(current_date + timedelta(days=requested_chunk_days - 1), end_dt)
            actual_chunk_days = (chunk_end_date - current_date).days + 1

            yield {
                "start_date": current_date,
                "end_date": chunk_end_date,
                "range_days": range_days,
                "actual_chunk_days": actual_chunk_days,
            }

            current_date = chunk_end_date + timedelta(days=1)

    def collect_chunked_endpoint_data(
        self,
        *,
        url: str,
        station_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        table_name: str,
        inplace: bool | None = None,
        logger: logging.Logger | None = None,
        date_filter_type: str | None = None,
        station_param_name: str | None = None,
        date_filter_param_name: str | None = None,
        search_date_param_name: str | None = None,
        range_param_name: str | None = None,
        extra_params: dict[str, Any] | None = None,
        ) -> None:
        """
        Collect endpoint data in date chunks while respecting API max-range limits;
        """
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        inplace = self.inplace if inplace is None else inplace
        active_logger = logger or self.logger
        date_filter_type = date_filter_type or self.date_filter_type
        station_param_name = station_param_name or self.station_param_name
        date_filter_param_name = date_filter_param_name or self.date_filter_param_name
        search_date_param_name = search_date_param_name or self.search_date_param_name
        range_param_name = range_param_name or self.range_param_name

        token = get_auth()
        if not token:
            active_logger.error("Failed to get authentication token")
            return

        headers = {"accept": "*/*", "Authorization": f"Bearer {token}"}
        all_dataframes: list[pd.DataFrame] = []
        chunk_number = 1

        # Chunks are generated to guarantee each call stays under API date limits;
        for chunk in self.iter_hidroweb_date_chunks(start_date, end_date):
            search_date = chunk["start_date"].strftime("%Y-%m-%d")
            range_days = chunk["range_days"]
            actual_chunk_days = chunk["actual_chunk_days"]

            params: dict[str, Any] = {
                station_param_name: station_code,
                date_filter_param_name: date_filter_type,
                search_date_param_name: search_date,
                range_param_name: range_days,
            }
            if extra_params:
                params.update(extra_params)

            active_logger.info(
                "Chunk %s: %s | range %s | days %s",
                chunk_number,
                search_date,
                range_days,
                actual_chunk_days,
            )
            response, headers = request_with_auth_retry(
                url=url,
                headers=headers,
                params=params,
                get_auth_token=get_auth,
                request_logger=active_logger,
            )

            if response.status_code != 200:
                active_logger.error("Chunk %s failed with status code: %s", chunk_number, response.status_code)
                active_logger.error("Response text: %s", response.text)
                chunk_number += 1
                continue

            payload = response.json()
            if "items" in payload and payload["items"]:
                all_dataframes.append(pd.DataFrame(payload["items"]))
                active_logger.info("Chunk %s collected successfully", chunk_number)
            else:
                active_logger.info("Chunk %s returned no items", chunk_number)
            chunk_number += 1

        if not all_dataframes:
            active_logger.warning("No data collected for endpoint '%s'", url)
            return

        combined_df = pd.concat(all_dataframes, ignore_index=True)
        db_handler = DBConnection()
        db_handler.write(combined_df, table_name, inplace=inplace)
        active_logger.info("Collected %s records and saved to '%s'", len(combined_df), table_name)

    # These endpoints require station and date-range params, so we route them through chunked collection;
    def HidrosatSerieDados(
        self,
        station_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        inplace: bool | None = None,
    ):
        self.collect_chunked_endpoint_data(
            url=BASE_URL+"HidrosatSerieDados/v1",
            extra_params={"Tipo Filtro Data": "DATA_LEITURA"},
            station_code=station_code,
            start_date=start_date,
            end_date=end_date,
            table_name="HidrosatSerieDados",
            inplace=inplace,
        )

    def HidroSerieVazao(
        self,
        station_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        inplace: bool | None = None,
    ):
        self.collect_chunked_endpoint_data(
            url=BASE_URL+"HidroSerieVazao/v1",
            station_code=station_code,
            start_date=start_date,
            end_date=end_date,
            table_name="HidroSerieVazao",
            inplace=inplace,
        )

    def HidroSerieSedimentos(
        self,
        station_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        inplace: bool | None = None,
    ):
        self.collect_chunked_endpoint_data(
            url=BASE_URL+"HidroSerieSedimentos/v1",
            station_code=station_code,
            start_date=start_date,
            end_date=end_date,
            table_name="HidroSerieSedimentos",
            inplace=inplace,
        )

    def HidroSerieResumoDescarga(
        self,
        station_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        inplace: bool | None = None,
    ):
        self.collect_chunked_endpoint_data(
            url=BASE_URL+"HidroSerieResumoDescarga/v1",
            station_code=station_code,
            start_date=start_date,
            end_date=end_date,
            table_name="HidroSerieResumoDescarga",
            inplace=inplace,
        )

    def HidroSerieQA(
        self,
        station_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        inplace: bool | None = None,
    ):
        self.collect_chunked_endpoint_data(
            url=BASE_URL+"HidroSerieQA/v1",
            station_code=station_code,
            start_date=start_date,
            end_date=end_date,
            table_name="HidroSerieQA",
            inplace=inplace,
        )

    def HidroSeriePerfilTransversal(
        self,
        station_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        inplace: bool | None = None,
    ):
        self.collect_chunked_endpoint_data(
            url=BASE_URL+"HidroSeriePerfilTransversal/v1",
            station_code=station_code,
            start_date=start_date,
            end_date=end_date,
            table_name="HidroSeriePerfilTransversal",
            inplace=inplace,
        )

    def HidroSerieGranulometria(
        self,
        station_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        inplace: bool | None = None,
    ):
        self.collect_chunked_endpoint_data(
            url=BASE_URL+"HidroSerieGranulometria/v1",
            station_code=station_code,
            start_date=start_date,
            end_date=end_date,
            table_name="HidroSerieGranulometria",
            inplace=inplace,
        )

    def HidroSerieCurvaDescarga(
        self,
        station_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        inplace: bool | None = None,
    ):
        self.collect_chunked_endpoint_data(
            url=BASE_URL+"HidroSerieCurvaDescarga/v1",
            station_code=station_code,
            start_date=start_date,
            end_date=end_date,
            table_name="HidroSerieCurvaDescarga",
            inplace=inplace,
        )

    def HidroSerieCotas(
        self,
        station_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        inplace: bool | None = None,
    ):
        self.collect_chunked_endpoint_data(
            url=BASE_URL+"HidroSerieCotas/v1",
            station_code=station_code,
            start_date=start_date,
            end_date=end_date,
            table_name="HidroSerieCotas",
            inplace=inplace,
        )

    def HidroSerieChuva(
        self,
        station_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        inplace: bool | None = None,
    ):
        self.collect_chunked_endpoint_data(
            url=BASE_URL+"HidroSerieChuva/v1",
            station_code=station_code,
            start_date=start_date,
            end_date=end_date,
            table_name="HidroSerieChuva",
            inplace=inplace,
        )

    def HidroinfoanaSerieTelemetricaDetalhada_v1(
        self,
        station_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        inplace: bool | None = None,
    ):
        self.collect_chunked_endpoint_data(
            url=BASE_URL+"HidroinfoanaSerieTelemetricaDetalhada/v1",
            station_code=station_code,
            start_date=start_date,
            end_date=end_date,
            table_name="HidroinfoanaSerieTelemetricaDetalhada_v1",
            inplace=inplace,
        )

    def HidroinfoanaSerieTelemetricaDetalhada_v2(
        self,
        station_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        inplace: bool | None = None,
    ):
        self.collect_chunked_endpoint_data(
            url=BASE_URL+"HidroinfoanaSerieTelemetricaDetalhada/v2",
            station_code=station_code,
            start_date=start_date,
            end_date=end_date,
            table_name="HidroinfoanaSerieTelemetricaDetalhada_v2",
            inplace=inplace,
        )

    def HidroinfoanaSerieTelemetricaAdotada_v1(
        self,
        station_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        inplace: bool | None = None,
    ):
        self.collect_chunked_endpoint_data(
            url=BASE_URL+"HidroinfoanaSerieTelemetricaAdotada/v1",
            station_code=station_code,
            start_date=start_date,
            end_date=end_date,
            table_name="HidroinfoanaSerieTelemetricaAdotada_v1",
            inplace=inplace,
        )

    def HidroinfoanaSerieTelemetricaAdotada_v2(
        self,
        station_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        inplace: bool | None = None,
    ):
        self.collect_chunked_endpoint_data(
            url=BASE_URL+"HidroinfoanaSerieTelemetricaAdotada/v2",
            station_code=station_code,
            start_date=start_date,
            end_date=end_date,
            table_name="HidroinfoanaSerieTelemetricaAdotada_v2",
            inplace=inplace,
        )


def get_chunk_endpoint_map(
    api: Chunk_Hidroweb_API,
) -> dict[str, Callable[[str, str | None, str | None, bool | None], None]]:
    """
    Return a stable name -> bound method mapping for chunked endpoints.
    """
    return {
        "HidrosatSerieDados": api.HidrosatSerieDados,
        "HidroSerieVazao": api.HidroSerieVazao,
        "HidroSerieSedimentos": api.HidroSerieSedimentos,
        "HidroSerieResumoDescarga": api.HidroSerieResumoDescarga,
        "HidroSerieQA": api.HidroSerieQA,
        "HidroSeriePerfilTransversal": api.HidroSeriePerfilTransversal,
        "HidroSerieGranulometria": api.HidroSerieGranulometria,
        "HidroSerieCurvaDescarga": api.HidroSerieCurvaDescarga,
        "HidroSerieCotas": api.HidroSerieCotas,
        "HidroSerieChuva": api.HidroSerieChuva,
        "HidroinfoanaSerieTelemetricaDetalhada_v1": api.HidroinfoanaSerieTelemetricaDetalhada_v1,
        "HidroinfoanaSerieTelemetricaDetalhada_v2": api.HidroinfoanaSerieTelemetricaDetalhada_v2,
        "HidroinfoanaSerieTelemetricaAdotada_v1": api.HidroinfoanaSerieTelemetricaAdotada_v1,
        "HidroinfoanaSerieTelemetricaAdotada_v2": api.HidroinfoanaSerieTelemetricaAdotada_v2,
    }


def normalize_station_codes(raw_station_codes: list[str]) -> list[str]:
    """
    Normalize station input supporting both repeated args and comma-separated values.
    """
    normalized_codes: list[str] = []
    for station_entry in raw_station_codes:
        split_codes = [chunk.strip() for chunk in station_entry.split(",") if chunk.strip()]
        normalized_codes.extend(split_codes)
    return normalized_codes


def build_chunk_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for endpoint_chunk.py.
    """
    parser = argparse.ArgumentParser(
        description="Execute chunked HidroWeb endpoints for one or many stations."
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=["all"],
        help="Chunk endpoint method names to run. Use 'all' to execute every chunk endpoint.",
    )
    parser.add_argument(
        "--stationcode",
        nargs="+",
        required=True,
        help="Station code list. Accepts space-separated values and/or comma-separated batches.",
    )
    parser.add_argument(
        "--start-date",
        default=START_DATE,
        help="Start date in yyyy-MM-dd format.",
    )
    parser.add_argument(
        "--end-date",
        default=END_DATE,
        help="End date in yyyy-MM-dd format.",
    )
    parser.add_argument(
        "--inplace",
        dest="inplace",
        action="store_true",
        help="Overwrite destination table when writing to database (default).",
    )
    parser.add_argument(
        "--no-inplace",
        dest="inplace",
        action="store_false",
        help="Append to destination table instead of replacing it.",
    )
    parser.set_defaults(inplace=True)
    return parser


def run_chunk_cli(args: argparse.Namespace) -> None:
    """
    Execute selected chunked endpoints for one or many stations.
    """
    ensure_duckdb_exists()
    api = Chunk_Hidroweb_API()
    api.inplace = args.inplace
    endpoint_map = get_chunk_endpoint_map(api)
    station_codes = normalize_station_codes(args.stationcode)

    if not station_codes:
        raise ValueError("At least one valid station code must be provided via --stationcode.")

    requested_endpoints = args.endpoints
    if "all" in requested_endpoints:
        selected_endpoint_names = list(endpoint_map.keys())
    else:
        invalid_endpoints = [name for name in requested_endpoints if name not in endpoint_map]
        if invalid_endpoints:
            raise ValueError(
                "Invalid chunk endpoints: %s. Valid options: %s"
                % (", ".join(invalid_endpoints), ", ".join(endpoint_map.keys()))
            )
        selected_endpoint_names = requested_endpoints

    # Iterate endpoint x station to keep run order explicit for logs and troubleshooting;
    for endpoint_name in selected_endpoint_names:
        endpoint_call = endpoint_map[endpoint_name]
        for station_code in station_codes:
            logger.info("Executing chunk endpoint: %s | station: %s", endpoint_name, station_code)
            endpoint_call(
                station_code=station_code,
                start_date=args.start_date,
                end_date=args.end_date,
                inplace=args.inplace,
            )


if __name__ == "__main__":
    cli_parser = build_chunk_parser()
    cli_args = cli_parser.parse_args()
    run_chunk_cli(cli_args)

# #================== Estações Guaíba ==================
# get_station_data(station_code="87450004", start_date="2014-07-01", end_date="2024-05-03", table_name="station_guaiba_1")
# get_station_data(station_code="87444000", start_date="2024-05-03", end_date=END_DATE, table_name="station_guaiba_2")

# #================== Estações Gravataí ==================
# get_station_data(station_code="87399000", start_date="2018-07-01", end_date=END_DATE, table_name="station_gravatai_1")

# #================== Estações Sinos ==================
# get_station_data(station_code="87382000", start_date="2018-07-01", end_date=END_DATE, table_name="station_sinos_1")
# get_station_data(station_code="87380000", start_date="2013-12-01", end_date=END_DATE, table_name="station_sinos_2")

# #================== Estações Taquari ==================
# get_station_data(station_code="86510000", start_date="2017-10-01", end_date=END_DATE, table_name="station_taquari_1")
# get_station_data(station_code="86720000", start_date="2008-12-01", end_date=END_DATE, table_name="station_taquari_2")

# #================== Estações Caí ==================
# get_station_data(station_code="87150000", start_date="2010-01-01", end_date=END_DATE, table_name="station_cai_1")
# get_station_data(station_code="87170000", start_date="2018-01-01", end_date=END_DATE, table_name="station_cai_2")

# #================== Estações Jacuí ==================
# get_station_data(station_code="85900000", start_date="2017-10-01", end_date=END_DATE, table_name="station_jacui_1")
# logger.info("All Done!")