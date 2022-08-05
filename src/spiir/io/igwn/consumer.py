"""Contains basic functionality to consume data from the IGWN Alert Kafka service."""
import logging
from pathlib import Path
from typing import Any, Optional

import toml
from igwn_alert import client
from ligo.gracedb.rest import GraceDb


logger = logging.getLogger(__name__)


class IGWNAlertConsumer:
    def __init__(
        self,
        id: Optional[str] = None,
        service_url: str = f"https://gracedb-playground.ligo.org/api/",
        out_dir: str = "out/results",
    ):
        self.id = id or type(self).__name__  # replace with process/node id?
        self.gracedb = None
        self.service_url: str | None = None
        if service_url is not None:
            self._setup_client(service_url)

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)

    def _setup_client(self, service_url: str):
        if self.gracedb is not None:
            self.gracedb.close()
        try:
            self.gracedb = GraceDb(service_url=service_url, reload_certificate=True)
            self.service_url = service_url
            logger.info(f"{self.id} initialised connection to {self.service_url}")
        except Exception as e:
            logger.warning(e)
      
    def process_alert(
        self,
        topic: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ):
        logger.debug(f"{self.id} doing nothing with payload from {topic}: {payload}.")
        pass


# TODO: Add docstring(s)
# TODO: Incorporate as method on IGWNAlertConsumer class
def run_igwn_alert_consumer(
    consumer: IGWNAlertConsumer,
    server: str = "kafka://kafka.scima.org/",
    group: str = "gracedb-playground",
    topics: list[str] = ["test_spiir"],
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
    logger.debug(
        ' '.join([f"{k}: {v}" for k, v in client_args.items() if k != "password"] + ".")
    )

    alert_client = client(**client_args)

    try:
        logger.debug(f"Listening to topics: {', '.join(topics)}.")
        alert_client.listen(consumer.process_alert, topics)

    except (KeyboardInterrupt, SystemExit):
        # Kill the client upon exiting the loop:
        logger.info(f"Disconnecting from: {server}")
        try:
            alert_client.disconnect()
        except:
            logger.info("Disconnected")
