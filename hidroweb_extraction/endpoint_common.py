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
class Common_Hidroweb_API():
    def __init__(self) -> None:
        self.token = get_auth()
        self.inplace = True
        self.logger = logger
        self.db_handler = DBConnection()

    def collect_endpoint_data(
        self,
        *,
        logger: logging.Logger | None = None,
        url: str,
        table_name: str,
        inplace: bool | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """
        Collect one-shot endpoint data and persist response items into DuckDB database;
        """
        active_logger = logger or self.logger
        inplace = self.inplace if inplace is None else inplace

        if not self.token:
            self.token = get_auth()
        if not self.token:
            active_logger.error("Failed to get authentication token")
            return

        headers = {"accept": "*/*", "Authorization": f"Bearer {self.token}"}
        response, updated_headers = request_with_auth_retry(
            url=url,
            headers=headers,
            params=params,
            get_auth_token=get_auth,
            request_logger=active_logger,
        )
        authorization_header = updated_headers.get("Authorization", "")
        if authorization_header.startswith("Bearer "):
            self.token = authorization_header.replace("Bearer ", "", 1)

        if response.status_code != 200:
            active_logger.error("Request failed with status code: %s", response.status_code)
            active_logger.error("Response text: %s", response.text)
            return

        payload = response.json()
        if "items" not in payload or not payload["items"]:
            active_logger.warning("No data items found in response for URL '%s'", url)
            return

        df = pd.DataFrame(payload["items"])
        self.db_handler.write(df, table_name, inplace=inplace)
        active_logger.info("Data collected from '%s' and saved to '%s'", url, table_name)

    # Endpoints:
    def HidrosatInventarioEstacoes(self):
        self.collect_endpoint_data(
            url=BASE_URL+"HidrosatInventarioEstacoes/v1",
            table_name="HidrosatInventarioEstacoes",
        )
    def HidroUF(self):
        self.collect_endpoint_data(
            url=BASE_URL+"HidroUF/v1",
            table_name="HidroUF",
        )
    def HidroSubBacia(self):
        self.collect_endpoint_data(
            url=BASE_URL+"HidroSubBacia/v1",
            table_name="HidroSubBacia",
        )
    def HidroRio(self):
        self.collect_endpoint_data(
            url=BASE_URL+"HidroRio/v1",
            table_name="HidroRio",
        )
    def HidroMunicipio(self):
        self.collect_endpoint_data(
            url=BASE_URL+"HidroMunicipio/v1",
            table_name="HidroMunicipio",
        )
    def HidroInventarioEstacoes(self):
        # Get all stations for all Federative regions;
        ufs = self.db_handler.run(query="SELECT DISTINCT Estado_Sigla FROM HidroUF").get("result", pd.DataFrame()).values
        for uf in ufs:
            self.collect_endpoint_data(
                url=BASE_URL+"HidroInventarioEstacoes/v1",
                params = {"Unidade Federativa": uf},
                table_name="HidroInventarioEstacoes",
                inplace=False
            )
    def HidroEntidade(self):
        self.collect_endpoint_data(
            url=BASE_URL+"HidroEntidade/v1",
            table_name="HidroEntidade",
        )
    def HidroBacia(self):
        self.collect_endpoint_data(
            url=BASE_URL+"HidroBacia/v1",
            table_name="HidroBacia",
        )


def get_common_endpoint_map(api: Common_Hidroweb_API) -> dict[str, Callable[[], None]]:
    """
    Return a stable name -> bound method mapping for common endpoints.
    """
    return {
        # "HidrosatInventarioEstacoes": api.HidrosatInventarioEstacoes, #BROKEN ENDPOINT
        "HidroUF": api.HidroUF,
        "HidroSubBacia": api.HidroSubBacia,
        "HidroRio": api.HidroRio,
        "HidroMunicipio": api.HidroMunicipio,
        "HidroInventarioEstacoes": api.HidroInventarioEstacoes,
        "HidroEntidade": api.HidroEntidade,
        "HidroBacia": api.HidroBacia,
    }


def build_common_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for endpoint_common.py.
    """
    parser = argparse.ArgumentParser(
        description="Execute one-shot HidroWeb endpoints and persist results."
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=["all"],
        help="Endpoint method names to run. Use 'all' to execute every common endpoint.",
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


def run_common_cli(args: argparse.Namespace) -> None:
    """
    Execute selected common endpoints from CLI arguments.
    """
    ensure_duckdb_exists()
    api = Common_Hidroweb_API()
    api.inplace = args.inplace
    endpoint_map = get_common_endpoint_map(api)

    requested_endpoints = args.endpoints
    if "all" in requested_endpoints:
        selected_endpoint_names = list(endpoint_map.keys())
    else:
        invalid_endpoints = [name for name in requested_endpoints if name not in endpoint_map]
        if invalid_endpoints:
            raise ValueError(
                "Invalid common endpoints: %s. Valid options: %s"
                % (", ".join(invalid_endpoints), ", ".join(endpoint_map.keys()))
            )
        selected_endpoint_names = requested_endpoints

    # Execute sequentially so logs and DB writes remain deterministic and easy to audit;
    for endpoint_name in selected_endpoint_names:
        logger.info("Executing common endpoint: %s", endpoint_name)
        endpoint_map[endpoint_name]()


if __name__ == "__main__":
    cli_parser = build_common_parser()
    cli_args = cli_parser.parse_args()
    run_common_cli(cli_args)