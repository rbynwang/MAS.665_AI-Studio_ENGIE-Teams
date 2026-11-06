# FlexGrid code structure

The source code isn't posted here yet. This page describes how it's organized.

## Folders

```text
flexgrid-agent/
├── flexgrid_core/   the agents: coordinator, homes, buildings, market, settlement
├── components/      Langflow components, one step of a flow each
├── flows/           builds the Langflow flows from the components
├── openclaw/        each agent's role, tools and skills
├── models/          trained forecast models
├── tools/           build data, train models, run simulations, check results
├── tests/           checks that live and offline runs agree
├── viz/             Control Room, household app and talk pages
├── agent/           test buyer agent
├── deploy/          edge device installation and services
└── mac/             coordinator machine installation, start and stop
```

## Main modules in `flexgrid_core/`

| Area | Modules | What they do |
|---|---|---|
| Coordinator | `coordinator.py`, `market.py`, `rt_market.py`, `community_forecast.py` | Run the community market, handle calls and gaps in real time, forecast community demand |
| Home node | `daemon.py`, `home_v2.py`, `fleet.py` | Run a home agent, make offers, deliver and report, run the simulated homes |
| Building node | `building.py` | Runs a large building through the same protocol as a home |
| Planning | `rt_control.py`, `rt_home.py`, `hvac.py` | Re-plan battery, car and cooling every 5 minutes within the owner's limits |
| Forecasting | `forecaster.py`, `treemodel.py` | Forecast home load and solar from trained models |
| AI layer | `brain.py`, `llm.py`, `agent_mcp.py` | Let the AI propose decisions, with code checking them, and give agents their tools |
| Messaging | `beckn.py`, `signing.py`, `town.py` | Build Beckn messages, sign them, send them over NANDA Town |
| Settlement | `settlement.py` | Measure delivery against a baseline and split payment |
| Device layer | `adapters/` | Give every device one interface. Only the simulated adapter works today |
| Test buyers | `utility.py`, `rt_buyer.py`, `buyer_agent.py`, `mcp_server.py` | Stand in for the buyer side until the link to it runs |
| Monitoring | `monitor.py`, `telemetry.py` | Feed the Control Room and check sensor readings |

Data center modules are still in the code but are out of scope and not used in the current plan.
