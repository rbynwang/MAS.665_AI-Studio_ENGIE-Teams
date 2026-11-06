# FlexGrid system architecture

FlexGrid is the sell side of the energy exchange: a community coordinator over home and large building nodes. This page shows the target design. The status table says which parts exist today.

## The system

```text
 Buyer side             awards, prices and calls come down
   |                    one pooled offer and signed readings go up
 Connection             still to agree between teams
   |
 Coordinator            one seller for the community: forecast, day-ahead clearing,
   |                    gap handling every 5 minutes, verification, payment split
   |
   |  signed messages over NANDA Town
   |
 Shared AI model        reasons for each node in turn
   |
 NODE, one per home or building, on an edge device
   Sends up             offers, signed readings, labelled answers, shortfall reports
   Agent state          jobs, logs and memory
   Policy engine        who may change what, deny by default
   Context MCP          schedule, comfort, device groups, device cards, tools
   Forecasts            load, solar, car and sellable power
   Controller           plans all devices every 5 minutes
   Safety guard         can overrule everything above it
   Drivers              real or simulated devices
   |
 Devices and people
```

## Status

| Part | Status |
|---|---|
| Community coordinator and day-ahead clearing | Built |
| Gap handling every 5 minutes | Built |
| Signed messages over NANDA Town | Built |
| Home agents, with the AI model on a small number of homes and rules on the rest | Built |
| Home forecasts for load, solar and car | Built |
| 5-minute home controller | Built |
| Settlement from signed meter readings | Built |
| Large building node | Built and tested, not live |
| Machine learning forecast for buildings | Planned |
| Policy engine, context MCP and agent memory | Planned |
| Dedicated safety guard | Planned. Hard limits are checked in the executor today |
| Labelled answers from each node | Planned |
| Real device drivers | Planned. Devices are simulated, and no real device is connected |
| Connection to the buyer side | Planned. Test buyers stand in until it runs |

## How an event works

1. A buyer places an order.
2. The coordinator calls for offers.
3. Each node's planner works out a safe offer, and its agent may lower it.
4. The coordinator clears the market and sends commitments.
5. Each node checks its share, and its agent may decline.
6. During the event, nodes re-plan every 5 minutes and sign their meter readings.
7. The coordinator covers shortfalls from its margin, then moves what's left to other nodes at a rising price, or warns the buyer.
8. Settlement pays on the meter readings, and the buyer verifies the result.

## Who decides what

The AI proposes and code decides. Code does all the arithmetic, and real-time control never waits for the AI model.

| Who | May do | May never do |
|---|---|---|
| Person | Set limits, say "I need it now" | Disable life safety |
| Node agent | Lower offers, choose among safe options, report | Raise a limit or drive a device directly |
| Controller | Plan within limits | Exceed a device's rated limits |
| Safety guard | Overrule anything, return to the normal schedule | Be switched off by software above it |
| Coordinator | Request capacity, move a gap | Command a device |

This table is the target design. It depends on the planned policy engine and safety guard.

## To agree with other teams

- Offer format
- Signed reading format
- Transport
- Which reference measures delivery
- One shared dataset
- How to catch a correctly signed offer that overstates what a node can deliver
