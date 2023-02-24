"""Module for the IGWNAlert Consumer that predicts p_astro and uploads to GraceDb."""

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..consumer import IGWNAlertConsumer

if TYPE_CHECKING:
    from ....search.p_astro.models import CompositeModel

logger = logging.getLogger(__name__)


class PAstroAlertConsumer(IGWNAlertConsumer):
    def __init__(
        self,
        model,
        out: str = "./out/",
        topics: List[str] = ["test_spiir"],
        group: str = "gracedb-playground",
        server: str = "kafka://kafka.scima.org/",
        id: Optional[str] = None,
        username: Optional[str] = None,
        credentials: Optional[str] = None,
        upload: bool = False,
        save_payload: bool = False,
    ):
        super().__init__(topics, group, server, id, username, credentials)
        self.model = model  # assumes p-astro model already loaded
        self.out_dir = Path(out)  # location to save results from process_alert
        self.gracedb = self._setup_gracedb_client(group)
        self.upload = upload

    def __exit__(self):
        """Close connections associated with Consumer object."""
        if self.gracedb is not None:
            self.gracedb.close()

    def __enter__(self):
        """Enables use within a with context block."""
        return self

    def __exit__(self):
        """Enables use within a with context block."""
        self.close()

    def close(self):
        """Closes all client connections."""
        if self.gracedb is not None:
            self.gracedb.close()
        super().close()

    @staticmethod
    def _write_json(data: Dict[str, Any], path: Path):
        """Write dictionary data to a JSON file."""
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open(mode="w") as f:
            f.write(json.dumps(data, indent=4))

    def _setup_gracedb_client(self, group: Optional[str] = None):
        """Instantiate connection to GraceDb via GraceDb client."""
        groups = {"gracedb", "gracedb-test", "gracedb-playground"}
        if group is not None and group not in groups:
            raise ValueError(f"gracedb must be one of {groups}, not '{group}'.")
            service_url = f"https://{group}.ligo.org/api/"
            return GraceDb(service_url=service_url, reload_certificate=True)

    def _get_data_from_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve data for p_astro model from the IGWNAlert payload."""
        try:
            data = payload["data"]["extra_attributes"]
        except Exception as exc:
            logger.debug(f"[{self.id}] No valid data retrieved from payload: {exc}.")
            return

        far = data["CoincInspiral"]["combined_far"]
        if far <= 0.0:
            logger.debug(f"[{self.id}] FAR is equal to 0. - skipping")
            return

        snr = data["CoincInspiral"]["snr"]
        mchirp = data["CoincInspiral"]["mchirp"]
        eff_dist = min(sngl["eff_distance"] for sngl in data["SingleInspiral"])

        return {"far": far, "snr": snr, "mchirp": mchirp, "eff_dist": eff_dist}

    def process_alert(
        self,
        topic: Optional[List[str]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ):
        """Process IGWN Alerts by retrieving coinc.xml files and computing p_astro."""
        runtime = time.perf_counter()

        # parse payload input from topic
        if payload is None:
            logger.debug(f"[{self.id}] Alert received from {topic} without a payload.")
            return
        elif not isinstance(payload, dict):
            try:
                payload = json.loads(payload)
            except Exception as exc:
                logger.debug(f"[{self.id}] Error loading {topic} JSON payload: {exc}.")
                return

        # only get the first alert for a given event - initial coinc.xml upload
        if payload.get("alert_type", None) != "new":
            return

        # get GraceDb id in database
        try:
            event_id = payload["uid"]
        except KeyError as exc:
            logger.debug(f"[{self.id}] No uid in {topic} payload: {payload}: {exc}.")
            return

        if self.save_payload:
            payload_file = self.out_dir / event_id / "payload.json"
            self._write_json(payload, payload_file)
            logger.info(f"[{self.id}] Uploading {gid} payload to {payload_file}.")

        # compute p_astro
        data = self._get_data_from_payload(data)
        p_astro = self.model.predict(**data)

        # create spiir.p_astro.json file and upload to GraceDb
        p_astro_file = event_dir / "spiir.p_astro.json"
        self._write_json(p_astro, p_astro_file)

        if self.upload:
            logger.info(f"[{self.id}] Uploading {gid} p_astro to {self.group}.")
            self.gracedb.writeLog(gid, "source probabilities", filename=p_astro_file)

        runtime = time.perf_counter() - runtime
        logger.debug(f"[{self.id}] {event_id} alert processed in {runtime:.4f}s.")
