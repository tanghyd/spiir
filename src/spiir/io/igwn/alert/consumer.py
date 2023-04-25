"""Contains basic functionality to consume data from the IGWN Alert Kafka service."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
from igwn_alert import client

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
        self.client = self._setup_igwn_alert_client(self.username, self.credentials)

    def __enter__(self):
        """Enables use within a with context block."""
        return self

    def __exit__(self, *args, **kwargs):
        """Enables use within a with context block."""
        self.close()

    def close(self):
        """Closes all client connections."""
        if self.client is not None:
            self.client.disconnect()

    def _setup_igwn_alert_client(self, username: str, credentials: str) -> client:
        """Instantiate IGWNAlert client connection."""
        # specify default SCiMMA auth.toml credentials path
        auth_fp = Path(credentials).expanduser()
        assert Path(auth_fp).is_file(), f"{auth_fp} is not a file."

        # prepare igwn alert client
        kwargs = {"server": self.server, "group": self.group, "authfile": str(auth_fp)}

        # load SCIMMA hop auth credentials from auth.toml file
        if username is not None:
            auth_data = toml.load(auth_fp)
            auth = [data for data in auth_data["auth"] if data["username"] == username]

            # handle ambiguous/duplicate usernames
            if len(auth) > 1:
                msg = f"[{self.id}] Ambiguous credentials for {username} in {auth_fp}"
                raise RuntimeError(msg)
            elif len(auth) == 0:
                msg = f"[{self.id}] No credentials found for {username} in {auth_fp}"
                raise RuntimeError(msg)
            else:
                logger.debug(
                    f"[{self.id}] Loading {username} credentials from {auth_fp}"
                )
                kwargs["username"] = auth[0]["username"]
                kwargs["password"] = auth[0]["password"]
        else:
            logger.debug(f"[{self.id}] Loading default credentials from {auth_fp}")

        return client(**kwargs)

    def process_alert(
        self,
        topic: Optional[List[str]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ):
        if payload is None:
            return
        logger.debug(f"[{self.id}] Received payload from {topic}.")

    def subscribe(self, topics: Optional[List[str]] = None):
        topics = topics or self.topics
        logger.debug(f"[{self.id}] Listening to topics: {', '.join(topics)}.")
        try:
            self.client.listen(self.process_alert, topics)

        except (KeyboardInterrupt, SystemExit):
            # Kill the client upon exiting the loop:
            logger.info(f"[{self.id}] Disconnecting from: {self.server}")
            try:
                self.close()
            except Exception:
                logger.info(f"[{self.id}] Disconnected")
