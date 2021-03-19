# 41X-simulation
Simulation For Object detection for occupancy system

### Installations required:

Requires:
- Poetry (https://python-poetry.org/docs/)

Postgres SQL DB storage
- `brew install postgresql`



Visualization
- `brew install grafana`
- use the [quick start guide](https://grafana.com/docs/grafana/latest/getting-started/getting-started/)

## Getting Started

### Start by launching the poetry shell

`poetry shell`

### Install required dependencies

`poetry install`

### Run the simulation 

`python idemographer-sim.py <detection_type>`

detection_type can be `image`, `video`, or `earthcam`



