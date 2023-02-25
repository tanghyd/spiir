import argparse
import logging
import time
from pathlib import Path
from typing import Optional, List, Union

import click
from ligo.gracedb.rest import GraceDb

from spiir.cli import click_logger_options
from spiir.logging import setup_logger

@click.command()
@click.argument("files", nargs=-1, type=click.Path())
@click.option("--pipeline", "-p", type=str, default="spiir")
@click.option("--group", "-g", type=str, default="Test")
@click.option("--search", "-s", type=str)
@click.option("--service-url", type=str, default="https://gracedb-playground.ligo.org/api/")
@click.option("--wait", type=float, default=0.5)
@click_logger_options
def main(
    files: List[str],
    pipeline: str = "spiir",
    group: str = "Test",
    search: Optional[str] = None,
    service_url: str = "https://gracedb-playground.ligo.org/api/",
    wait: float = 0.5,
    log_level: int = logging.WARNING,
    log_file: Optional[Union[str, Path]] = None,
):
    logger = setup_logger(Path(__file__).stem, log_level, log_file)

    if wait < 0:
        raise ValueError("wait must be a float greater than or equal to 0.")

    with GraceDb(service_url=service_url) as client:

        if search is not None:
            assert search in client.searches

        for fp in files:
            if Path(fp).is_file():
                try:
                    response = client.create_event(group, pipeline,  fp, search=search)
                    logger.debug(f"Sent {fp} with response {response.status_code}")
                except Exception as exc:
                    try:
                        # if response content is accessible
                        logger.debug(f"Sent {fp} with response {response.content}")
                    except Exception:
                        pass
                    logger.warning(exc)  # log original exception

                time.sleep(wait)

            else:
                logger.info(f"{fp} does not exist. Skipping...")


if __name__ == "__main__":
    main()