#!/usr/bin/env python
import logging
from pathlib import Path
from typing import List, Optional, Union

import click

from spiir.io.igwn.alert.consumers import PAstroAlertConsumer
from spiir.search.p_astro.models import CompositeModel
from spiir.cli import click_logger_options
from spiir.logging import setup_logger

@click.argument("signal-config", type=click.Path(exists=True))
@click.argument("source-config", type=click.Path(exists=True))
@click.option("--out", type=click.Path(file_okay=False))
@click.option("--topics", "-t", multiple=True)
@click.option("--group", "-g", type=str, default="gracedb-playground")
@click.option("--server", "-s", type=str, default="kafka://kafka.scima.org/")
@click.option("--id", type=str)
@click.option("--username", "-u", type=str)
@click.option("--credentials", type=str)
@click.option("--upload", type=bool, is_flag=True)
@click.option("--save-payload", type=bool, is_flag=True)
@click_logger_options
def main(
    signal_config: str,
    source_config: str,
    out: str = "./out/",
    topics: List[str] = ["test_spiir"],
    group: str = "gracedb-playground",
    server: str = "kafka://kafka.scima.org/",
    id: Optional[str] = None,
    username: Optional[str] = None,
    credentials: Optional[str] = None,
    upload: bool = False,
    save_payload: bool = False,
    log_level: int = logging.WARNING,
    log_file: Optional[Union[str, Path]] = None,
):
    # configure logging
    setup_logger("spiir", log_level, log_file)

    # load p_astro composite model from .pkl files
    model = CompositeModel()
    model.load(signal_config, source_config)

    # instantiate and run P Astro consumer
    with PAstroAlertConsumer(
        model=model,
        out=out,
        topics=topics,
        group=group,
        server=server,
        id=id,
        username=username,
        credentials=credentials,
        upload=upload,
        save_payload=save_payload,
    ) as consumer:
        consumer.run()

if __name__ == '__main__':
    main()
