"""Module for the IGWNAlert Consumer that predicts p_astro and uploads to GraceDb."""

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ligo.gracedb.rest import GraceDb

from ..consumer import IGWNAlertConsumer

if TYPE_CHECKING:
    from ....search.p_astro.models import CompositeModel

logger = logging.getLogger(__name__)


class PAstroAlertConsumer(IGWNAlertConsumer):
    def __init__(
        self,
        model: "CompositeModel",
        out: str = "./out/",
        topics: List[str] = ["test_spiir"],
        group: str = "gracedb-playground",
        server: str = "kafka://kafka.scima.org/",
        id: Optional[str] = None,
        username: Optional[str] = None,
        credentials: Optional[str] = None,
        upload_gracedb: bool = False,
        save_payload: bool = False,
    ):
        super().__init__(topics, group, server, id, username, credentials)
        self.model = model  # assumes p-astro model already loaded
        self.out_dir = Path(out)  # location to save results from process_alert
        self.save_payload = save_payload

        self.upload_gracedb = upload_gracedb
        self.gracedb = self._setup_gracedb_client(group) if upload_gracedb else None

    def __enter__(self):
        """Enables use within a with context block."""
        return self

    def __exit__(self, *args, **kwargs):
        """Enables use within a with context block."""
        self.close()

    def close(self):
        """Closes all client connections."""
        if self.gracedb is not None:
            self.gracedb.close()
        super().close()

    def _write_json(self, data: Dict[str, Any], path: Path):
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

    def _get_data_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
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

        # get GraceDb id in database to label payload and p_astro files
        try:
            event_id = payload["uid"]
        except KeyError as exc:
            logger.debug(f"[{self.id}] No uid in {topic} payload: {payload}: {exc}.")
            return

        # retrieve data from payload
        if self.save_payload:
            payload_fp = self.out_dir / event_id / "payload.json"
            self._write_json(payload, payload_fp)
            logger.info(f"[{self.id}] Saving {event_id} payload file to: {payload_fp}.")

        data = self._get_data_from_payload(payload)
        if data is None:
            return

        # compute p_astro
        p_astro = self.model.predict(**data)

        # create spiir.p_astro.json file and upload to GraceDb
        p_astro_fp = self.out_dir / event_id / "spiir.p_astro.json"
        self._write_json(p_astro, p_astro_fp)

        if self.upload_gracedb:
            logger.info(f"[{self.id}] Uploading {event_id} p_astro to {self.group}.")
            self.gracedb.writeLog(event_id, "source probabilities", filename=p_astro_fp)

        runtime = time.perf_counter() - runtime
        logger.debug(f"[{self.id}] Alert for {event_id} processed in {runtime:.4f}s.")
