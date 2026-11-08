# FlexGrid

**Every load in this folder is simulated.** Homes and buildings replay a dataset, the homes are synthetic, and no real device or meter is involved. [DATA_NOTICE.md](DATA_NOTICE.md) says what each dataset is.

FlexGrid is a market for demand flexibility. On a hot evening the grid needs people to use less power for a few hours. Homes and buildings can do that: a battery discharges, a car waits to charge, a building lets its cooling drift, a dryer runs later. One home's cut is too small to sell, so FlexGrid pools many into one sale.

Each home and building is a seller node, and each node is an agent: a model decides its offer, its answer to a contract, its plan when the sale opens and whenever something changes, and how it reads its owner's notes. Code's rules are one input the agent sees and may take, change or drop. A guard checks every device call against the person's limits, and a rule decides a point only when the model fails, which is counted. A coordinator pools the offers and contracts with buyers who have ordered a cut for a window, with a deposit. The nodes deliver, each sends signed meter readings to the coordinator and the buyers, and the buyers pay on those readings. Every message is a signed [Beckn](https://becknprotocol.io/) envelope, and the messages travel over [Nanda Town](https://github.com/projnanda/nandatown)'s HTTP mailbox.

This is course work for MIT AI Studio, Fall 2026 (Homework 8). It is not a product, and it has not run on a real grid.

## What is here

| Path | What |
|---|---|
| `flexgrid_core/` | The package: the coordinator, the buyers, the home and building nodes and their agents, the guard, the market, settlement, signing, and the Town client |
| `tools/sim_event.py`, `tools/market_day.py`, `tools/scorecard.py` | The offline simulator: the same code plays a day in one process, with an in-memory bus and a virtual clock, and no Town. Every node runs on rules there |
| `data_year_t1b/` | A small data pack: 8 homes and 3 buildings, 1 to 28 Jan and 1 Jun to 3 Jul 2026, every 15 minutes |

The live deployment, the trained models, the user interfaces, the test suite and the benchmark are not in this folder.

## Install

Python 3.11 or newer. From this folder, in a clone of this repository:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e .
```

Then play one market day offline. It takes a few seconds and prints what was sold, delivered and paid:

```bash
export FLEXGRID_CONFIG_DIR=$(mktemp -d) FLEXGRID_DATASET=ma_year_t1b FG_MARKET=2026 FG_BATTERY_MODE=backup FG_DEVICES=hpwh,pool,dryer,dish,v2h
python tools/market_day.py 2026-06-29 --homes 8
```

`FLEXGRID_CONFIG_DIR` is where FlexGrid keeps a node's keys and settings, so point it at a folder of its own. Nothing of that folder belongs in this repository.

## How the Nanda Town check uses it

The check lives in Nanda Town's repository, in `examples/flexgrid/`, and FlexGrid's code does not. Set `FLEXGRID_HOME` to this folder in a clone of this repository, and `FLEXGRID_BASELINE_HOME` to the same folder in a worktree of the baseline tag. With the clone in a folder called `flexgrid`, from the folder that holds it:

```bash
git -C flexgrid worktree add ../flexgrid-6nov flexgrid-hw8-baseline-6nov
F=teams/robyn-community-coordination/code
export FLEXGRID_HOME=$PWD/flexgrid/$F FLEXGRID_BASELINE_HOME=$PWD/flexgrid-6nov/$F
```

Then, from a Nanda Town checkout, `python examples/flexgrid/run_check.py --cases normal_rules` plays one day with every node on rules, and `--cases normal --model URL --model-name NAME` plays it with every node an agent, on an Ollama-style model.

The check copies `flexgrid_core/` and the three tools to a temporary folder, links `data_year_t1b/`, and writes nothing to the checkout. For each case it does three things:

1. It plays the day offline with `tools/sim_event.py`, every node on rules, to find what the sellers should be paid on rules.
2. It starts Town and FlexGrid's own processes: the coordinator, two buyers, a fleet of 8 homes, and 3 buildings. It plays the same day over Town's mailbox, with each of them a Town participant of its own. The clock waits for the agents: at each 15-minute step every clock holds until every agent has answered. That is a test and demo device: a real evening doesn't wait.
3. It compares Town's records with FlexGrid's ledgers. A rules day is held to the offline day to the cent. An agent's day is held to its own receipts: each node's offer on Town's record is the amount its agent chose, and an answer below the rule's amount carries a reason its owner or manager heard.

## Two versions

| Version | Where | What it is |
|---|---|---|
| 6 Nov 2026 baseline | tag `flexgrid-hw8-baseline-6nov` | The code as it ran on 6 Nov 2026. The check's failure case runs on it: batch ids repeat on a second play, Town refuses them, and the sellers go unpaid |
| Current | tag `flexgrid-hw8-recorded`, and this folder on `main` | Every node an agent, the clock that waits for them, and the fixes since 6 Nov: a message's id says which message it is, the same within a play and new across plays, a refused send is counted, nodes keep their state on disk through a crash, and readings are checked against what a meter can do |

Both have the same data pack.

## Limits

- Every load is simulated, and no real device, meter or grid is involved.
- The wholesale price columns are a flat placeholder, not market prices.
- The check's community is small: 8 homes and 3 buildings, one day per case, on one machine, over loopback.
- An agent's day needs a model, and a model's answers can differ between runs. The check asks at temperature 0 with a fixed seed, which makes most of them repeat. The trained forecast models are not here, so forecasts use a simple regression on temperature.
- The waiting clock exists only in a replay. A real heat wave won't pause for the agents.
- The offline reference is FlexGrid's own code. Agreement on a rules day shows Town changed nothing, not that the settlement rules are right.
- A signature says who signed, not that the number is true. A meter that lies inside its limits is paid.
- The control endpoints such as `/demo/start` ask for a token only once a node's config folder holds one (`python -m flexgrid_core.doors token`). The check runs with none. This code is for a local test, not for an open network.
- Many comments describe the live system this code came from, and they name files that aren't in this repository.

## Licence

The code is under the Apache License 2.0: see [LICENSE](LICENSE). The data pack is not covered by it: see [DATA_NOTICE.md](DATA_NOTICE.md).

`settlement.deg_settle` follows the demand-flex settlement policy published in [beckn/DEG](https://github.com/beckn/DEG), and `beckn.py` uses that project's message vocabulary. That repository is under CC BY-NC-SA 4.0.

Weather data by [Open-Meteo.com](https://open-meteo.com/). Other credits are in the notice.
