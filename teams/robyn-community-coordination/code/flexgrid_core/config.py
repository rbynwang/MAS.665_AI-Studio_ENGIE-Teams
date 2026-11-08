"""Agent configuration: ~/.config/flexgrid/agent.json over built-in defaults.

The two Jetsons run identical code; only this file differs (agent name, brand,
replay profile, battery size). Secrets (Town join token, Langflow API key, signing
key) live in ~/.config/flexgrid/secrets/ and never in the repo.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(os.environ.get("FLEXGRID_CONFIG_DIR", str(Path.home() / ".config" / "flexgrid")))
REPO_DIR = Path(__file__).resolve().parents[1]

# Which dataset every agent replays: il2023 (data/, the team's 2023 Illinois dataset) or ma2025 (data_ma/, the
# 2025-26 Massachusetts community built by tools/make_ma_dataset.py). Chosen by $FLEXGRID_DATASET, else the repo's
# DATASET file (deployed with the code, so every machine agrees), else il2023.
DATA_DIRS = {"il2023": "data", "ma2025": "data_ma", "ma_year": "data_year", "ma_year_t1": "data_year_t1", "ma_year_t2": "data_year_t2", "ma_year_t3": "data_year_t3",
             "ma_year_t1b": "data_year_t1b", "ma_year_t2b": "data_year_t2b", "ma_year_t3b": "data_year_t3b"}   # the same homes: true plug-in times, the full sunlight forecast


def _dataset_name() -> str:
    env = os.environ.get("FLEXGRID_DATASET", "").strip()
    if env:
        return env
    f = REPO_DIR / "DATASET"
    return f.read_text().strip() if f.exists() else "il2023"


DATASET = _dataset_name()
DATA_DIR = REPO_DIR / DATA_DIRS.get(DATASET, "data")
IL2023_META: dict[str, Any] = {
    "name": "il2023", "label": "Illinois 2023 (team dataset)", "tz_offset_h": -5, "tz_label": "CDT",
    "range": ["2023-01-21", "2023-12-30"], "replay_start": "2023-04-22 00:00", "events_file": "grid_events_2023.json",
    "default_event": "2023-08-24", "holidays": [], "agent_defaults": {}, "profile_alias": {}, "site_labels": {},
    "price_label": "MISO real-time price, Illinois hub, 2023 (dataset)",
    "program_label": "Ameren Illinois Peak Time Rewards: $0.75 per kWh reduced",
    "community": {"name": "20 Illinois counties"}, "models": "models/forecast_2025_26",
    "datacenters": {"year": 2023, "region": "region_2023.csv.gz", "solar": "community_solar_2023.csv.gz",
                    "events": "large_load_events_2023.json", "grid": "MISO", "operator": "MISO",
                    "price": "MISO day-ahead price, Illinois hub, 2023 (dataset)", "data_word": "the 2023 dataset"}}
_META: dict[str, Any] | None = None


def dataset_meta() -> dict[str, Any]:
    """The active dataset's settings (data_ma/dataset.json; Illinois defaults when the file is absent)."""
    global _META
    if _META is None:
        f = DATA_DIR / "dataset.json"
        meta = dict(IL2023_META)
        if f.exists():
            meta.update(json.loads(f.read_text()))
        _META = meta
    return _META

DEFAULTS: dict[str, Any] = {
    "agent_name": "home-tesla",          # NANDA Town participant name
    "role": "home",                      # home | building (a building node's daemon, building.py) | coordinator
    "brand": "tesla",                    # tesla | enphase (which real adapter to use later)
    "adapter": "simulated",              # simulated (dataset replay) | tesla | enphase
    "profile": "tesla_calhoun",          # data/home_<profile>.csv.gz + data/ev_<profile>.csv
    "site_label": "Tesla home, Calhoun County",
    "battery_kwh": 13.5,
    "battery_kw": 5.0,
    "reserve_pct": 20.0,
    "soc_max_pct": 100.0,
    "soc0_pct": 55.0,
    "ev_max_kw": 7.2,
    "export_limit_kw": 10.0,
    "eta": 0.95,                         # one-way battery efficiency
    "control": "auto",                   # auto | advisory (advisory: recommend, homeowner applies)
    "replay_start": "2023-04-22 00:00",  # local time in the dataset
    "replay_step_seconds": 60,           # wall seconds per 15-minute dataset step
    "replay_loop_days": 1,               # loop the replay after N days (0 = run on)
    "town_url": "http://127.0.0.1:8477",
    "town_run_id": "",
    "coordinator_name": "coordinator",
    "langflow_url": "http://127.0.0.1:7860",
    "flows": {"H1": "", "H2": ""},
    "daemon_host": "127.0.0.1",
    "daemon_port": 7870,
    "max_dispatch_minutes": 120,
    "min_seconds_between_dispatches": 0,
    # coordinator only (the Mac)
    "coordinator_port": 7871,            # coordinator daemon API, 127.0.0.1 only
    "feeder_limit_kw": 8.0,              # most the community may push in one direction at once (grid safety check)
    "report_max_age_s": 150,             # older flex.reports are skipped (e.g. backlog while the coordinator was off)
    "dispatch_ttl_s": 120,               # a dispatch request expires if not acted on within this many wall seconds
    # v2 (Beckn demand-flex market; see market.py, utility.py, fleet.py)
    "v1_dispatch": False,                # True = also run the v1 hourly rule dispatch (off: v2 contracts run events)
    "utility_name": "utility",           # NANDA Town name of the grid buyer ("Utility or ENGIE" mock)
    "clear_margin": 0.10,                # homes are asked for 10 % more than the community promises the utility (the
                                         # coordinator's AI agent picks 5-30 % per event when llm_enabled)
    "clear_lead_minutes": 180,           # clear at the latest this long before the utility's commit deadline
    "coordinator_fee_share": 0.0,        # share of the event revenue the coordinator keeps (0 in the demo)
    "send_workers": 8,                   # parallel Town sends when the coordinator broadcasts to 100 homes
    "inbox_workers": 4,                  # coordinator threads claiming from its Town inbox (bursts of ~100 messages)
    "catchup_workers": 16,               # ...at least this many, so a backlog after downtime clears faster
    "max_message_age_s": 900,            # a Beckn message without its own ttl is dropped this many seconds after sending
    "v1_town_reports": False,            # homes: also send the v1 flex.report through Town every minute (off: v2 market)
    "discover_retry_wall_seconds": 30,   # coordinator: ask the utility again at most this often (real time)
    "utility_retry_minutes": 30,         # coordinator: select/init/confirm again after 30, 60, 120 ... replay minutes
    "utility_retry_wall_seconds": 60,    # ...and after 60, 120, 240 ... real seconds (a fast demo clock)
    "utility_retries": 5,                # ...then give up (also at the utility's deadline)
    "readings_wait_minutes": 120,        # coordinator: dataset minutes after the window to wait for every home's readings
    "utility_settle_wait_minutes": 30,   # utility: settle this many replay minutes after "readings are in", even if some are missing
    "clock_profile": "tesla_calhoun",    # the replay trace whose clock the coordinator and utility follow
    "fleet_port": 7872,                  # flexgrid-fleet status API, 127.0.0.1 only
    "fleet_poll_workers": 8,             # threads that take turns checking the 98 fleet inboxes (no long-polls)
    "fleet_poll_seconds": 5.0,           # how often each fleet inbox is checked (each check is a Town write: keep low)
    "utility_port": 7873,                # flexgrid-utility status API, 127.0.0.1 only
    "fleet_opt_out": {},                 # {home: ["2023-08-24", ...]} homeowner opt-out days for replay homes
    "buildings": [],                     # the Mac (coordinator, utility, Control Room): building nodes that take part,
                                         # by name in data_ma/buildings.json (e.g. ["bld-library"]); [] = homes only
    "forecast": "temp_regression",       # home load forecast for offers: temp_regression | naive (previous 3 days)
    "langflow_keep_hours": 48,           # Langflow run history (traces, Playground sessions) kept this long
    "rebound_tolerance": 0.0,            # evening peak may exceed the no-event forecast peak by this share (0 = not)
    # the AI agents (brain.py): the LLM proposes, the agent's own code decides. Off by default: the 98 simulated
    # homes, the offline simulations and the tests run on rules
    "llm_enabled": False,                # this machine's agent thinks with the local LLM (a Jetson home, the coordinator)
    "llm_runtime": "openclaw",           # openclaw: through this machine's OpenClaw gateway (its agent, skills, MCP tools),
                                         # with Ollama straight as the stand-in when the gateway is down | ollama: straight
    "llm_url": "http://127.0.0.1:11434", # Ollama on the Mac; on a Jetson the tunnel to it: http://127.0.0.1:11500
    "llm_model": None,                   # None = the first model of llm.PREFERRED the server has
    "openclaw_url": "http://127.0.0.1:18789",   # this machine's OpenClaw gateway (token: secrets/openclaw_token)
    "openclaw_agent": None,              # the OpenClaw agent id (None: "home" on a Jetson, "coordinator" on the Mac)
    "llm_offer_budget_s": 40,            # most a home's offer decision may take (then: the plan's offer)
    "llm_accept_budget_s": 45,           # ...a home's yes or no to its commitment (then: yes, the guard found it safe);
                                         # one model answers every agent in turn, so a yes or no can wait behind
                                         # another agent's decision (at 20 s the Tesla home's fell back in live runs)
    "llm_margin_budget_s": 40,           # ...the coordinator's margin (then: clear_margin)
    "llm_chat_budget_s": 120,            # ...an answer to the homeowner in the phone app
}


def load() -> dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULTS))
    meta = dataset_meta()
    cfg.update(meta.get("agent_defaults", {}))
    path = CONFIG_DIR / "agent.json"
    if path.exists():
        for k, v in json.loads(path.read_text()).items():
            # agent.json often holds every default, saved by an older version: there the dataset's own default
            # (e.g. forecast = trained for Massachusetts) wins; a value someone changed on purpose is kept
            if DATASET != "il2023" and k in meta.get("agent_defaults", {}) and v == DEFAULTS.get(k):
                continue
            cfg[k] = v
    if DATASET != "il2023":
        lo, hi = meta["range"]
        if not lo <= str(cfg.get("replay_start", ""))[:10] <= hi:     # a start left over from another dataset
            cfg["replay_start"] = meta["replay_start"]
        if cfg.get("agent_name") in meta.get("site_labels", {}):
            cfg["site_label"] = meta["site_labels"][cfg["agent_name"]]
    return cfg


def save(cfg: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "agent.json").write_text(json.dumps(cfg, indent=2))


def secret(name: str) -> str:
    p = CONFIG_DIR / "secrets" / name
    return p.read_text().strip() if p.exists() else ""


def write_secret(name: str, value: str) -> None:
    d = CONFIG_DIR / "secrets"
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    p = d / name
    p.write_text(value.strip() + "\n")
    os.chmod(p, 0o600)


def data_path(name: str) -> Path:
    """A file of the active dataset. What a dataset does not have falls back to the Illinois 2023 data: the
    data-center study (data/datacenters, community_solar_2023, region_2023) exists only there."""
    p = DATA_DIR / name
    if not p.exists() and DATA_DIR.name != "data":
        q = REPO_DIR / "data" / name
        if q.exists():
            return q
    return p


def events_file() -> Path:
    """The grid-event catalog the mock utility publishes."""
    return data_path(dataset_meta()["events_file"])


def state_dir() -> Path:
    d = CONFIG_DIR / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d
