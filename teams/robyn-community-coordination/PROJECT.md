# Robyn: community energy coordination

- **Project name:** FlexGrid
- **People:** Robyn
- **Last updated:** 2026-11-06

## What we are building

FlexGrid is the sell side of the energy exchange. A community coordinator works with home and large building nodes to offer their flexible energy use to buyers.

Grid operators, utilities and data centers are out of scope.

## Status

**Working now:**

All of this runs on simulated devices. No real device is connected yet.

- Community coordinator, day-ahead market and gap handling every 5 minutes
- Home agents with signed messages over NANDA Town
- 5-minute planning for each home's battery, car and cooling
- Trained load, solar and car forecasting models for homes
- Settlement from signed meter readings
- A large building node, built and tested but not live

**In progress:**

- Fleshing out the home and large building nodes

**Planned:**

- Edge devices in homes that connect through MCP to understand context behind the meter
- A machine learning forecast for buildings
- A link to the buyer side

## How to run or see it

The source code isn't posted yet. See the [system architecture](docs/ARCHITECTURE.md) and [code structure](docs/CODE_STRUCTURE.md).

## What we take in and what we give out

**Inputs:** Home and building energy data, forecasts, and limits set by owners

**Outputs:** Community flexibility offers for buyers

## What we need from other teams

- **Buyer side (Cooper):** agree the offer format, delivery reference, transport and a shared dataset
- **Trust layer (Walmy):** a way to catch a correctly signed offer that overstates what a node can deliver

## Links

- [System architecture](docs/ARCHITECTURE.md)
- [Code structure](docs/CODE_STRUCTURE.md)
