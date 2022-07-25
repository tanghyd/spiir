import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import toml
from igwn_alert import client
from ligo.gracedb.rest import GraceDb

from spiir.search.p_astro.mass_contour import MassContourEstimator

logger = logging.getLogger(__name__)


def run_igwn_alert_consumer(
    server: str = "kafka://kafka.scima.org/",
    group: str = "gracedb-playground",
    topics: list[str] = ["test_spiir"],
    outdir: str = "out/results",
    username: Optional[str] = None,
    credentials: Optional[str] = None,
):

    # specify default auth.toml credentials path
    credentials = credentials or "~/.config/hop/auth.toml"
    auth_fp = Path(credentials).expanduser()
    assert Path(auth_fp).is_file(), f"{auth_fp} does not exist"

    # prepare igwn alert client
    client_args = {"server": server, "group": group, "authfile": str(auth_fp)}

    if username is not None:
        # load SCIMMA hop auth credentials from auth.toml file
        auth_data = toml.load(auth_fp)
        auth = [data for data in auth_data["auth"] if data["username"] == username]

        # handle ambiguous/duplicate usernames
        if len(auth) > 1:
            raise RuntimeError(f"Ambiguous credentials for {username} in {auth_fp}")
        elif len(auth) == 0:
            raise RuntimeError(f"No credentials found for {username} in {auth_fp}")
        else:
            logger.debug(f"Loading {username} credentials from {auth_fp}")
            client_args["username"] = auth[0]["username"]
            client_args["password"] = auth[0]["password"]
    else:
        logger.debug(f"Loading default credentials from {auth_fp}")

    # Initialize the client sesion
    logger.debug(client_args)
    alert_client = client(**client_args)

    service_url = f"https://{group}.ligo.org/api/"
    listener = IGWNAlertConsumer(out_dir=outdir, service_url=service_url)

    try:
        alert_client.listen(listener.process_alert, topics)

    except (KeyboardInterrupt, SystemExit):
        # Kill the client upon exiting the loop:
        logger.info(f"Disconnecting from: {server}")
        try:
            alert_client.disconnect()
        except:
            logger.info("Disconnected")


class IGWNAlertConsumer:
    def __init__(
        self,
        id: str = "IGWNAlertListener",
        service_url: str = f"https://gracedb-playground.ligo.org/api/",
        out_dir: str = "out/results",
    ):
        self.id = id  # replace with process/node id?

        # gracedb connection
        self.gracedb = None
        self.service_url: str | None = None
        if service_url is not None:
            self._setup_client(service_url)

        # output directory for results
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)

        # instantiate pastro model - to do: load from config + save coeffs with preds
        # coefficients = {"m0": 0.01, "a0": 0.75, "b0": -0.322, "b1": -0.516}  # pycbc
        coefficients = {"m0": 0.01, "a0": 0.76, "b0": -0.685, "b1": 0.467}  # spiir
        self.pastro = MassContourEstimator(coefficients)

        logger.info(f"Initialized {self.id}.")

    def _setup_client(self, service_url: str):
        if self.gracedb is not None:
            self.gracedb.close()
        try:
            self.gracedb = GraceDb(service_url=service_url)
            self.service_url = service_url
            logger.info(f"Initialised GraceDB connection at {self.service_url}")
        except Exception as e:
            logger.warning(e)

    def save_json(self, data: dict[str, Any], file_path: Path):
        with Path(file_path).open(mode="w") as f:
            f.write(json.dumps(data, indent=4))
            logger.debug(f"Saved {str(file_path)} to disk")

    def upload_pastro(self, graceid: str, probs: dict[str, Any]):
        assert self.gracedb is not None
        for key in ("BNS", "NSBH", "BBH", "MassGap"):  # "Terrestrial
            assert key in probs, f"{key} not present in {list(probs.keys())}"

        try:
            self.gracedb.createVOEvent(graceid, voevent_type="preliminary", **probs)
            logger.debug(f"{graceid} pastro uploaded to GraceDB")
        except Exception as e:
            logger.warning(e)

    def process_alert(
        self,
        topic: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ):
        # to do: check optional input parameters in igwn_alert repo
        if payload is not None:
            logger.info(f"{self.id} recieved an alert from topic {topic}")

            # extract relevant alert data from payload
            gid = payload["uid"]
            logger.info(f"Alert came from event ID: {gid}")

            # to do: pastro.json uploads may get confused with coinc.xml uploads
            # we need to make sure we are filtering for coinc events only

            data = payload["data"]["extra_attributes"]
            mchirp = data["CoincInspiral"]["mchirp"]
            snr = data["CoincInspiral"]["snr"]
            eff_dist = min(sngl["eff_distance"] for sngl in data["SingleInspiral"])

            # compute pastro prediction
            probs = self.pastro(mchirp, snr, eff_dist)
            logger.debug(f"{gid} pastro: {probs}")

            # upload pastro to gracedb
            self.upload_pastro(gid, probs)

            # save to disk; this is ~90% of the ~0.86s runtime - this should be async?
            alert_path = self.out_dir / gid
            alert_path.mkdir(exist_ok=True, parents=True)
            self.save_json(payload, alert_path / "payload.json")
            self.save_json(probs, alert_path / "pastro.json")

            # to do: refactor so we don't estimate pastro twice
            self.pastro.plot(
                mchirp,
                snr,
                eff_dist,
                suptitle=r"SPIIR Relative $P_{astro}$ Estimate for " + f"{gid}",
                outfile=alert_path / "mass_contour.png",  # save to disk
            )
            logger.debug(f"Saved {str(alert_path / 'mass_contour.png')} to disk")
        else:
            logger.warn(f"Alert received but payload = None; topic = {topic}")

    def __exit__(self):
        self.gracedb.close()