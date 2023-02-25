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
from spiir.io.igwn.alert.consumers import PAstroAlertConsumer
from spiir.search.p_astro.models import CompositeModel


def parse_cli_arguments() -> argparse.Namespace:
    """Parses command line arguments for running the p_astro model with IGWN Alert."""

    parser = argparse.ArgumentParser(
        description="Run the SPIIR p_astro IGWNAlertConsumer."
    )
    parser.add_argument("signal_config", help="Path to pre-trained signal model .pkl.")
    parser.add_argument("source_config", help="Path to pre-trained source model .pkl.")
    parser.add_argument(
        "--out",
        type=str,
        default="./eresults/",
        help="Output directory for input payload and output p_astro.",
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
    signal_config: str,
    source_config: str,
    out: str = "./results/",
    topics: List[str] = ["test_spiir"],
    group: str = "gracedb-playground",
    server: str = "kafka://kafka.scimma.org/",
    username: Optional[str] = 'daniel.tang-aef798b8',
    credentials: Optional[str] = None,
    log_level: int = logging.WARNING,
    log_file: Optional[Union[str, Path]] = None,
):
    setup_logger("spiir", log_level, log_file)

    model = CompositeModel()
    model.load(signal_config, source_config)

    with PAstroAlertConsumer(
        model,
        out="./results/",
        topics=topics,
        group=group,
        server=server,
        username=username,
        credentials=credentials,
        upload_gracedb=False,
        save_payload=True,
    ) as consumer:
        consumer.subscribe()
    

if __name__ == '__main__':
    args = parse_cli_arguments()
    main(**args.__dict__)
