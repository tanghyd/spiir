import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..consumer import IGWNAlertConsumer

if TYPE_CHECKING:
    from ....search.p_astro.models import CompositeModel


logger = logging.getLogger(__name__)


class PAstroCompositeModelConsumer(IGWNAlertConsumer):
    def __init__(
        self,
        model: "CompositeModel",
        id: Optional[str] = None,
        service_url: str = f"https://gracedb-playground.ligo.org/api/",
        out_dir: str = "p_astro/results",
    ):
        id = id or type(self).__name__
        super().__init__(id, service_url, out_dir)
        self.model = model  # assumes the model is already initialised

    def save_json(self, data: dict[str, Any], file_path: Path, indent: int = 4):
        with Path(file_path).open(mode="w") as f:
            f.write(json.dumps(data, indent=indent))

    def process_alert(
        self,
        topic: Optional[List[str]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ):
        # to do: check optional input parameters in igwn_alert repo
        if payload is not None:
            duration = time.perf_counter()
            if not isinstance(payload, dict):
                try:
                    payload = json.loads(payload)
                except Exception as exc:
                    logger.warning(f"Error parsing {topic} payload {payload}: {exc}.")
                    return

            if payload.get("alert_type", None) != "new":
                return

            try:
                gid = payload["uid"]
            except KeyError as exc:
                logger.debug(f"No uid found in {topic} payload: {payload}: {exc}.")
                return

            event_dir = self.out_dir / gid
            event_dir.mkdir(exist_ok=True, parents=True)
            self.save_json(payload, event_dir / "payload.json")

            # retrieve p_astro model input data from igwn alert payload
            try:
                data = payload["data"]["extra_attributes"]
            except Exception as exc:
                logger.debug(f"No valid data retrieved from {gid}: {exc}.")
                return

            far = data["CoincInspiral"]["combined_far"]
            if far <= 0.0:
                logger.debug(f"{gid} FAR is equal to 0. - skipping")
                return

            snr = data["CoincInspiral"]["snr"]
            mchirp = data["CoincInspiral"]["mchirp"]
            eff_dist = min(sngl["eff_distance"] for sngl in data["SingleInspiral"])

            # run p_astro model prediction
            runtime = time.perf_counter()
            p_astro = self.model.predict(far, snr, mchirp, eff_dist)
            runtime = time.perf_counter() - runtime
            logger.debug(f"{gid} p_astro predicted in {runtime:.4f}s.")

            # create p_astro.json file and upload to GraceDb
            runtime = time.perf_counter()
            event_file = event_dir / "p_astro.json"
            self.save_json(p_astro, event_file)
            if self.gracedb is not None:
                self.gracedb.writeLog(gid, "source probabilities", filename=event_file)
            runtime = time.perf_counter() - runtime
            logger.debug(f"{gid} p_astro.json uploaded to GraceDB in {runtime:.4f}s.")

            duration = time.perf_counter() - duration
            logger.debug(f"{gid} total processing time was {duration:.4f}s.")
        else:
            logger.warning(f"Alert received but payload = None; topic = {topic}")

    def __exit__(self):
        self.gracedb.close()
