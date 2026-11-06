# Robin: community energy coordination

- **Project name:** FlexGrid
- **People:** Robin
- **Last updated:** 2026-11-06

## What we are building

FlexGrid is the sell side of the energy exchange. A community coordinator works with home and large building nodes to offer their flexible energy use to buyers.

Grid operators, utilities and data centers are out of scope.

## Status

**Working now:**

- Trained load, solar and car forecasting models for homes
- Agent tools for home and building nodes

**In progress:**

- Fleshing out the home and large building nodes

**Planned:**

- Edge devices in homes that connect through MCP to understand context behind the meter
- A machine learning forecast for buildings
- A link to the buyer side

## How to run or see it

Code and demo links to be added.

## What we take in and what we give out

**Inputs:** Home and building energy data, forecasts, and limits set by owners

**Outputs:** Community flexibility offers for buyers

## What we need from other teams

- **Buyer side (Cooper):** agree the offer format, delivery reference, transport and a shared dataset
- **Trust layer (Walmy):** a way to catch a correctly signed offer that overstates what a node can deliver

## Links
