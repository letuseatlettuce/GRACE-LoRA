# Workflows

## GRACE LoRA: working principle
GRACE stays frozen. For each organic element, a low-rank add-on B·A is added to the 12
`reducing_` coefficient tensors (I1, I2, rho). Training data mixes teacher-labelled molecules
with DFT materials (replay), so the add-ons learn molecules without forgetting materials.

```mermaid
flowchart LR
  subgraph data["Training data"]
    T["GRACE-OFF teacher<br/>labels organic molecules"] --> MIX["Training mix"]
    M["DFT materials (MPtrj) with organic elements<br/>replay, so nothing is forgotten"] --> MIX
  end
  subgraph student["Student: GRACE foundation model"]
    W0["Frozen weights W₀<br/>21.1 M parameters, 89 elements"] --> W["Merged weights<br/>W = W₀ + (α/r)·B·A"]
    AD["LoRA add-ons B·A, rank 8<br/>one per organic element:<br/>H C N O F P S Cl Br I"] --> W
  end
  MIX --> P["Predict energies and forces"]
  W --> P
  P --> L["Loss against the labels"]
  L -->|"gradient dL/dW"| CR["Chain rule<br/>dL/dA and dL/dB"]
  CR -->|"update only A and B"| AD
  W --> OUT["Specialised GRACE<br/>5 MB adapter; other elements bit-identical"]
```

## The skill: agentic optimisation
The agent reviews settings with the user, picks a teacher, sets up, then runs experiments in a
loop: a hypothesis, a training run, monitor checks, a ledger entry and a decision, until the
budget runs out or progress stops. Then it validates the best model and writes the report.
Optional features (dashed or marked optional) are off by default.

```mermaid
flowchart TD
  U["User: make GRACE better on my chemistry"] --> HW["Hardware check<br/>sysinfo + platform notes"]
  HW --> S["Settings review<br/>defaults, costs, optional features; user confirms"]
  S --> TE{"Teacher"}
  TE -->|default| GO["GRACE-OFF"]
  TE -->|"name, path or link"| OWN["User's GRACE model"]
  TE -->|unsure| REC["Ask for elements;<br/>recommend teacher + replay data"]
  GO --> SETUP
  OWN --> SETUP
  REC --> SETUP
  SETUP["Setup: environment, verified downloads,<br/>smoke test, prepare data"] --> MODE{"Mode"}
  MODE -->|optimise| H
  subgraph loop["Autonomous loop: repeats until the time budget or experiment cap is reached"]
    direction TB
    H["Hypothesis and rationale<br/>(predicted metrics, optional)"] --> EX["Experiment: train LoRA<br/>on teacher + replay data"]
    EX --> MON["Monitors: organics, materials,<br/>organic-element crystals"]
    MON --> LED["Ledger: config, metrics,<br/>feasibility, prediction score"]
    LED --> DEC{"Budget left and<br/>still improving?"}
  end
  DEC -->|yes| AGAIN(["Next experiment:<br/>back to Hypothesis ↺"])
  EX -. "on failure, optional" .-> HEAL["Self-heal: retry with<br/>safer settings"]
  LED -. optional .-> CRIT["Critic agent<br/>reviews the result"]
  LED -. optional .-> CK["Cockpit: local page or<br/>published artifact"]
  DEC -->|"no: stop"| VAL["Validate best vs base and teacher<br/>organics + materials groups A, B, C"]
  MODE -->|"validation only"| VAL
  VAL --> REP["Report: heatmap, parity plots,<br/>trade-off, decision log"]
```
