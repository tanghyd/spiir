""""An application to listen to igwn-alert using the hop-client based API."""

import argparse
import logging
from pathlib import Path

import spiir.io.igwn
from spiir.logging import configure_logger
from spiir.search import p_astro

VALID_GRACEDB_GROUPS = ["gracedb-playground"]

logger = logging.getLogger(Path(__file__).stem)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments used for training p_astro model(s)."""

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
        "-c",
        "--credentials",
        type=str,
        default="~/.config/hop/auth.toml",
        help="Location of auth.toml credentials file",
    )
    parser.add_argument(
        "-u",
        "--username",
        type=str,
        help="Username for SCIMMA hop authentication credentials in auth.toml",
    )
    parser.add_argument(
        "--fgmc",
        default="./models/fgmc.pkl",
        type=Path,
        help="Path to FGMC model state file.",
    )
    parser.add_argument(
        "--mchirp-area",
        default="./models/mchirp_area.pkl",
        type=Path,
        help="Path to ChirpMassAreaModel state file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default="./results",
        help="Output directory to store results.",
    )
    parser.add_argument(
        "--logging-config", type=Path, help="Optional path to logging file."
    )
    parser.add_argument(
        "--log-dir", default="./logs", type=Path, help="Optional path to logging file."
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # if args.logging_config is not None:
    #     logging.config.fileConfig(args.logging_config)
    #     logger = logging.getLogger("spiir")
    #     # manually configure imported packages that overwhelm logs
    #     # logging.getLogger("matplotlib").setLevel(logging.INFO)  # change to >= INFO
    #     logger.info(f"Loaded logging configuration from {args.logging_config}.")
    # else:

    log_file = None
    if args.log_dir is not None:
        args.log_dir.mkdir(exist_ok=True)
        log_file = args.log_dir / Path(__file__).stem
    logger = configure_logger(logger, level=args.log_level, file=log_file)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.group not in VALID_GRACEDB_GROUPS:
        raise ValueError(f"group {args.group} must be one of {VALID_GRACEDB_GROUPS}.")

    # load trained p_astro models
    if not args.fgmc.is_file():
        raise FileNotFoundError(f"FGMC model state file {args.fgmc} not found.")

    fgmc_model = p_astro.models.TwoComponentModel()
    fgmc_model.load(args.fgmc)
    logger.info(f"Loaded {fgmc_model} from {args.fgmc}.")

    if not args.mchirp_area.is_file():
        raise FileNotFoundError(f"ChirpMassArea model {args.mchirp_area} not found.")
    mchirp_area_model = p_astro.mchirp_area.ChirpMassAreaModel()
    mchirp_area_model.loadl(args.mchirp_area)
    logger.info(f"Loaded {mchirp_area_model} from {args.mchirp_area_model}.")

    # instantiate IGWN Alert consumer with the trained p_astro model to process alerts
    consumer = spiir.io.igwn.consumers.PAstroCompositeModelConsumer(
        model=p_astro.models.CompositeModel(fgmc_model, mchirp_area_model),
        service_url=f"https://{args.group}.ligo.org/api/",
        out_dir=args.out_dir,
    )

    spiir.io.igwn.run_igwn_alert_consumer(
        consumer=consumer,
        server=args.server,
        group=args.group,
        topics=args.topics,
        username=args.username,
        credentials=args.credentials,
    )
