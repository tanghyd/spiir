"""Contains basic functionality to consume data from the IGWN Alert Kafka service."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
from igwn_alert import client
from ligo.gracedb.rest import GraceDb

logger = logging.getLogger(__name__)


class IGWNAlertConsumer:
    def __init__(
        self,
        topics: List[str] = ["test_spiir"],
        group: str = "gracedb-playground",
        server: str = "kafka://kafka.scima.org/",
        id: Optional[str] = None,
        username: Optional[str] = None,
        credentials: Optional[str] = None,
    ):
        self.id = id or type(self).__name__
        self.topics = topics
        self.group = group
        self.server = server
        self.credentials = credentials or "~/.config/hop/auth.toml"
        self.username = username

        self.client = self.get_client()

    def get_client(self) -> client:
        """Instantiate IGWNAlert client connection."""
        # specify default SCiMMA auth.toml credentials path
        auth_fp = Path(credentials).expanduser()
        assert Path(auth_fp).is_file(), f"{auth_fp} is not a file."

        # load SCIMMA hop auth credentials from auth.toml file
        if username is not None:
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

        # prepare igwn alert client
        kwargs = {"server": self.server, "group": self.group, "authfile": str(auth_fp)}
        return client(**kwargs)

    def process_alert(self, topic, payload):
        logger.debug(f"[{self.id} {topic}: Received payload: {payload}.")
        pass

    def run(self, topics: Optional[List[str]] = None):
        topics = topics or self.topics
        logger.debug(f"Listening to topics: {', '.join(topics)}.")
        try:
            self.client.listen(self.process_alert, topics)

        except (KeyboardInterrupt, SystemExit):
            # Kill the client upon exiting the loop:
            logger.info(f"Disconnecting from: {server}")
            try:
                alert_client.disconnect()
            except Exception:
                logger.info("Disconnected")
