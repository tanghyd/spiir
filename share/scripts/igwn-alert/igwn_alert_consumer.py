#!/usr/bin/env python

import argparse
import json
import logging
import time
import toml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from igwn_alert import client

from spiir.logging import setup_logger
from spiir.io.igwn.alert import IGWNAlertConsumer


def parse_cli_arguments() -> argparse.Namespace:
    """Parses command line arguments used for running the basic IGWNAlertConsumer."""

    parser = argparse.ArgumentParser(
        description="Run the SPIIR p_astro IGWNAlertConsumer."
    )
    parser.add_argument(
        "-s",
        "--server",
        type=str,
        default="kafka://kafka.scimma.org/",
        help="URL of hop-client server to stream topics from.",
    )
    parser.add_argument(
        "-g",
        "--group",
        type=str,
        default="gracedb-playground",
        help="Name of GraceDB group",
    )
    parser.add_argument(
        "-t",
        "--topics",
        type=str,
        nargs="+",
        default=["test_spiir"],
        help="IGWN Kafka topics for the listener to subscribe to.",
    )
    parser.add_argument(
        "-u",
        "--username",
        type=str,
        help="Username for SCIMMA hop authentication credentials in auth.toml",
    )
    parser.add_argument(
        "-c",
        "--credentials",
        type=str,
        default="~/.config/hop/auth.toml",
        help="Location of auth.toml credentials file",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_const",
        dest="log_level",
        const=logging.DEBUG,
        default=logging.WARNING,
        help="Display all developer debug logging statements",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        dest="log_level",
        const=logging.INFO,
        help="Set logging level to INFO and display progress and information",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Specify location to output log file",
    )
    return parser.parse_args()


def main(
    topics: List[str] = ["test_spiir"],
    group: str = "gracedb-playground",
    server: str = "kafka://kafka.scimma.org/",
    username: Optional[str] = None,
    credentials: Optional[str] = None,
    log_level: int = logging.WARNING,
    log_file: Optional[Union[str, Path]] = None,
):
    setup_logger("spiir", log_level, log_file)  # configures logging level for spiir pkg

    # with context allows consumer to set up and tear down state if required
    with IGWNAlertConsumer(
        topics=topics,
        group=group,
        server=server,
        username=username,
        credentials=credentials,
    ) as consumer:
        consumer.subscribe()  # listen to IGWNAlert Kafka topics on loop

if __name__ == '__main__':
    kwargs = parse_cli_arguments().__dict__  # get keyword arguments from CLI
    main(**kwargs)  # run script
