import argparse
import logging
import time
from pathlib import Path

from ligo.gracedb.rest import GraceDb

from spiir.logging import configure_logger

logger = logging.getLogger(Path(__file__).stem)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the SPIIR p(astro) IGWNAlert Listener."
    )
    parser.add_argument(
        "--dir",
        type=str,
        # default="data/pastro/tests/",
        help="Directory where coinc.xml files are stored.",
    )
    parser.add_argument(
        "-f",
        "--files",
        type=str,
        nargs="+",
        help="File name of coinc.xml stored in dir for GraceDb event upload.",
    )
    parser.add_argument(
        "-s",
        "--search",
        type=str,
        help="Search parameter for client.createEvent.",
    )
    parser.add_argument(
        "-g",
        "--group",
        default="Test",
        type=str,
        help="Group parameter for client.createEvent.",
    )
    parser.add_argument(
        "-p",
        "--pipeline",
        default="spiir",
        type=str,
        help="Pipeline parameter for client.createEvent.",
    )
    parser.add_argument(
        "-w",
        "--wait",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls.",
    )
    parser.add_argument(
        "--service_url",
        type=str,
        default="https://gracedb-playground.ligo.org/api/",
        help="GraceDB Client service URL.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
        help="Display all developer debug logging statements",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
        help="Set logging level to INFO and display progress and information",
    )
    args = parser.parse_args()

    logger = configure_logger(logger, level=args.log_level)

    if args.wait < 0:
        raise ValueError("wait arg must be greater than or equal to 0.")

    if args.files is not None:
        logger.debug(args.files)
        file_dir = Path(args.dir) if args.dir else Path(".")
        file_paths = [file_dir / Path(f) for f in args.files]
    else:
        share_dir = Path(__file__).parent.parent.parent
        file_paths = list((share_dir / "data" / "pipeline" / "coinc").glob("*.xml"))

    with GraceDb(service_url=args.service_url) as client:

        if args.search:
            assert args.search in client.searches

        for fp in file_paths:
            if fp.is_file():
                try:
                    response = client.createEvent(
                        args.group, args.pipeline, str(fp), search=args.search
                    )
                    logger.debug(f"Sent {fp} with response {response.status_code}")
                except Exception as exc:
                    try:
                        # if response content is accessible
                        logger.debug(f"Sent {fp} with response {response.content}")
                    except Exception:
                        pass
                    logger.warning(exc)  # log original exception

                time.sleep(args.wait)

            else:
                logger.info(f"{fp} does not exist. Skipping...")
