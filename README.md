<div align="center">

# ENTITY GRID

<b>The low voltage grid has no map. I built one out of the smart meters that are already there.</b>

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Challenge](https://img.shields.io/badge/Yuva%20Yodha%202026-Grid%20Reliability-blue)](#what-entity-grid-is)
[![Python](https://img.shields.io/badge/python-3.12-green)](#quick-start)
[![Tests](https://img.shields.io/badge/tests-14%20passing-brightgreen)](#quick-start)
[![Hardware](https://img.shields.io/badge/new%20hardware-none-red)](#what-makes-it-different)
[![Connectivity](https://img.shields.io/badge/connectivity-100%25%20vs%2069.1%25%20ledger-orange)](#evidence-at-a-glance)

[quick start](#quick-start) / [what it is](#what-entity-grid-is) / [evidence](#evidence-at-a-glance) / [how it works](#how-the-topology-learner-works) / [limitations](#known-limitations)

</div>

## Quick start

Any machine with Python 3.12. Nothing else.

```bash
git clone https://github.com/kkjjkamal123/Schneider-Electric---yuva-yodha-tech.git
cd Schneider-Electric---yuva-yodha-tech
pip install -r requirements.txt
./run.sh
```

Windows PowerShell uses `.\run.ps1` instead. Either one generates the dataset, runs the pipeline and starts the server. Then open:

* `http://127.0.0.1:8000/` for the project page
* `http://127.0.0.1:8000/dashboard` for the operator console
* `http://127.0.0.1:8000/docs` for the API

Step by step if you would rather watch each stage:

```bash
cd backend
python -m entitygrid.sim.generate   # build the synthetic AMI dataset, about 15 s
python -m entitygrid.pipeline       # run all four pillars, about 10 s
python -m pytest tests/ -q          # 14 tests
python -m uvicorn entitygrid.api.main:app --port 8000
```

## What ENTITY GRID is

Below the distribution transformer, an Indian DISCOM is close to blind. Roughly one consumer in three is recorded against the wrong transformer or the wrong phase. Transformers fail without warning. When a low voltage feeder trips, crews still find the fault by driving the line.

Meanwhile RDSS is putting a smart meter on every connection in the country, and that data is used almost entirely for billing.

ENTITY GRID reads the same data differently. It recovers the connectivity of the low voltage network from voltage alone, then uses that map to score transformer condition, locate faults and manage the voltage rise that rooftop solar causes. It is software only. No new meters, no gateways, no site survey.

I validated it on a simulated network I wrote myself, solved with a four wire backward forward sweep, because no DISCOM was going to hand me a month of live AMI data for a hackathon. The truth is known exactly and no model is allowed to read it.

## What makes it different

| Decision | What ENTITY GRID does |
|---|---|
| Topology first | Everything downstream is expressed against the connectivity it learned, not the utility ledger. Feeding the later stages a ledger that is 31% wrong puts that error into every result that follows. |
| Learns rather than assumes | Conventional outage management needs an accurate network model and instrumented MV feeders. Indian LV networks have neither, which is why those deployments underperform here. Topology is treated as the unknown. |
| Neutral is modelled | The power flow solves three phases plus neutral. Consumers are single phase and unevenly spread, so the neutral carries the unbalance, and that coupling is the physical mechanism the topology learner exploits. Ignore the neutral and the signal does not exist. |
| Impedance, not load | Health tracks estimated ohms against each asset's own baseline. Overload raises current and drops voltage without changing the impedance between busbar and consumer. Degradation raises it. The two are separable only if you measure the right thing. |
| Two detectors, not one | A bad joint with three consumers behind it moves the transformer average by nothing. A second pass at meter level catches it, and names the group of consumers sitting behind the fault. |
| Unmeasurable is reported | An indicator whose noise exceeds its own signal is surfaced as unassessable rather than passed as healthy. Silent gaps are how monitoring systems lose a control room's trust. |
| Fits the existing stack | Designed to correct and enrich an ADMS, not replace one. Nobody has to rip anything out to adopt it. |

## Evidence at a glance

Every number below is graded against ground truth that the models never see. A test in the suite fails the build if the topology learner so much as opens a truth file.

| Claim | Evidence | Boundary |
|---|---|---|
| **Connectivity goes from 69.1% to 100%** | 486 consumers on 12 transformers, 30 days of 15 minute data. The utility ledger has 150 records wrong. All 150 are found and corrected, transformer and phase, from voltage alone. | Synthetic data. Published field results for meter to transformer mapping sit around 80%, and phase identification on real Irish AMI data reports about 93%. Expect the field number to land nearer those than mine. |
| **It is usable after two days** | Full accuracy from day two, 97.5% from a single day. The sensitivity sweep runs the whole learner again at 1, 2, 3, 5, 7, 14 and 30 days. | A cold start still needs those two days. Health baselines need seven before any alert is credible. |
| **It survives losing reads** | 99.2% joint accuracy with an extra 15% of reads blanked at random, 88.3% at 30%. | Blanking is uniform random. Real AMI loses reads in correlated bursts, which is harder. |
| **All three degrading transformers caught, no false alarms** | Mean 9 days of warning before the modelled failure point. Zero healthy transformers flagged. | Three degradation events, all of them neutral joint failures. No winding failure appears in this dataset, so that path is exercised but not proven. |
| **Segment detection finds what the transformer average hides** | One of the three faults has only 2 consumers of 45 downstream. It is invisible at transformer level and the meter level pass still finds it, 4 days ahead. | 28 of 486 meters are flagged. Within a flagged group, purity varies. The transformer attribution is right every time; the exact member list is not always. |
| **Faults located in about a second** | 6 of 6 outages detected, 100% precision, 89% recall, roughly 1 second from the first last gasp message. Each one is bracketed between the deepest consumer still lit and the shallowest one dark. | Recall is capped by the 14% of last gasp messages I deliberately drop to model RF collisions. Depth ordering correlates 0.58 with true distance, so the bracket narrows a search, it is not a fix. |
| **The winding indicator is honestly unusable here** | Busbar referenced winding estimates have day to day scatter larger than their own level, so they are reported as unassessable. 11 indicator and asset pairs are in that state. | This is a real gap, not a feature. Fixing it needs neighbour transformer regression rather than a fleet median reference, which is the next thing I would build. |

## How the topology learner works

This is the one idea everything else rests on.

A meter's voltage is dominated by things that say nothing about where it sits. The substation's own swing, tap changer steps, the daily load cycle of the whole 11 kV feeder. Every meter sees those at the same moment, so a correlation on raw voltage is near 1.0 for every pair and separates nothing.

Two transforms fix it. Work on `v[t] - v[t-1]` to kill slow drift, then subtract the cross sectional median at each interval to kill whatever the entire network saw at once. What survives is the local drop across the impedance a meter shares with its transformer.

On the reference dataset that leaves:

| Meter pair | Correlation of residual increments |
|---|---|
| Same transformer, same phase | **0.81** |
| Same transformer, different phase | -0.09 |
| Different transformer | -0.01 |

So the natural cluster is the transformer and phase pair, not the transformer. Meters sharing a phase conductor share its voltage drop almost exactly. Meters on different phases of the same transformer are coupled only weakly, and with opposite sign, through the neutral.

Agglomerative clustering recovers those 36 groups with an adjusted Rand index of 1.000. Matching each group's mean signature against per phase busbar telemetry then labels it with a real asset ID, so the output is named transformers and phases rather than anonymous cluster numbers.

## The four pillars

| | Pillar | What it produces |
|---|---|---|
| 1 | Topology | Meter to transformer to phase, with a confidence score and an explicit field check list |
| 2 | Health | Estimated impedance per transformer and per segment, drift against own baseline, days to failure |
| 3 | Fault location | Outage bracketed in ohms from the transformer, with the boundary consumers named |
| 4 | Voltage | Excursions split into reverse flow driven and load driven, with volt var setpoints for the first |

## Repository guide

```
backend/entitygrid/
  config.py         simulation and grid parameters
  io.py             the one place the dataset is read
  pipeline.py       end to end run, writes data/processed/results.json
  sim/
    network.py      radial LV feeders, ground truth, the corrupted ledger
    profiles.py     domestic, commercial and agricultural demand, rooftop PV
    powerflow.py    vectorised four wire backward forward sweep
    events.py       transformer degradation and LV faults
    generate.py     writes the dataset an AMI head end would produce
  topology/         pillar 1, features, learner, scoring, sensitivity sweep
  health/           pillar 2, features, transformer trend test, segment test
  faultloc/         pillar 3, depth estimation, event grouping, localisation
  voltvar/          pillar 4, excursion detection, volt var allocation
  api/main.py       FastAPI, JSON endpoints and both pages
backend/tests/      14 tests, including guards on the central claims
frontend/
  landing.html      project page
  index.html        operator console
```

## Design notes

The power flow is written, not imported. Radial LV at high R over X ratios is exactly where a backward forward sweep beats Newton Raphson, and modelling the neutral explicitly is what makes the topology signal exist at all. It also keeps the dependency list at seven packages.

The whole month solves at once. The sweeps loop over nodes, which number in the dozens, while every operation is vectorised across time, which numbers in the thousands. Twelve transformer months solve in about nine seconds.

Health is a trend test, not a classifier. No utility has a labelled history of transformer failures, so supervised learning is not available in the field. Every indicator is an impedance in ohms that a distribution engineer can argue with.

## Known limitations

Better you read these from me than find them yourself.

| | Limitation | Detail |
|---|---|---|
| | **The validation is synthetic** | The physics is genuine and the confounders are modelled, including PV, tap changes, meter noise, comms gaps and a wrong ledger. It is still not a live AMI feed. Field accuracy will be lower and I have not claimed otherwise anywhere in the project. |
| | **The winding health path is unproven** | All three modelled degradations are neutral joint failures. The winding indicator exists, runs and is correctly reported as unassessable on this data, but nothing here demonstrates it catching a winding fault. |
| | **Peer comparison fails for feeder wide faults** | The segment detector compares a meter against peers on the same transformer and phase. When a fault sits close enough to the busbar that nearly every peer is downstream of it too, the reference is contaminated and the deviation cancels. The transformer level neutral indicator covers that case, which is why both exist. |
| | **Electrical depth is approximate** | Within transformer rank correlation against true distance has a median of 0.58 and a minimum below zero. The fault bracket narrows the search. It does not point at a pole. |
| | **Missing reads are modelled as random** | Real head ends lose data in correlated bursts, by feeder or by collector. Uniform random blanking is the easier version of that problem. |
| | **One network, one seed** | Twelve transformers and 486 consumers, generated from a single seed. The sensitivity sweep varies observation window and missing rate, not network topology or consumer mix. |

## Acknowledgements

The method draws on published work on data driven phase identification, in particular the finding that voltage correlation deteriorates as a customer's own load rises, and that a transformer makes a better correlation reference than a three phase customer. Where the literature and my results disagree, the literature is measured on real feeders and mine is not.

## License

Apache 2.0. See [LICENSE](LICENSE).
