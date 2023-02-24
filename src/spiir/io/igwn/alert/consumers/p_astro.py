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
        gracedb: Optional[str] = None,
        results_dir: str = "./results",
        topics: List[str] = ["test_spiir"],
        group: str = "gracedb-playground",
        server: str = "kafka://kafka.scima.org/",
        id: Optional[str] = None,
        username: Optional[str] = None,
        credentials: Optional[str] = None,
        save_payload: bool = True,
    ):
        super().__init__(topics, group, server, id, username, credentials)

        # create p_astro model and specify output directory
        self.model = model  # assumes p-astro model already loaded
        self.out_dir = Path(out_dir)  # location to save results from process_alert
        self.gracedb = self._setup_gracedb(gracedb)  # connect to GraceDb client

    def __exit__(self):
        """Close connections associated with Consumer object."""
        if self.gracedb is not None:
            self.gracedb.close()

    @staticmethod
    def _write_json(data: Dict[str, Any], path: Path):
        """Write dictionary data to a JSON file."""
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open(mode="w") as f:
            f.write(json.dumps(data, indent=4))

    def _setup_gracedb(self, gracedb: Optional[str] = None):
        """Instantiate connection to GraceDb via GraceDb client."""
        services = {"gracedb", "gracedb-test", "gracedb-playground"}
        if gracedb is not None and gracedb not in services:
            raise ValueError(f"gracedb must be one of {services}, not '{gracedb}'.")
            service_url = f"https://{gracedb}.ligo.org/api/"
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
            logger.debug(f"No uid found in {topic} payload: {payload}: {exc}.")
            return

        if self.save_payload:
            self._write_json(payload, self.out_dir / event_id / "payload.json")

        # compute p_astro
        data = self._get_data_from_payload(data)
        p_astro = self.model.predict(**data)

        # create spiir.p_astro.json file and upload to GraceDb
        p_astro_file = event_dir / "spiir.p_astro.json"
        self._write_json(p_astro, p_astro_file)

        if self.gracedb is not None:
            self.gracedb.writeLog(gid, "source probabilities", filename=p_astro_file)

        runtime = time.perf_counter() - runtime
        logger.debug(f"{event_id} alert processed in {runtime:.4f}s.")
