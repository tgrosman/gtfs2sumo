<div align="center">

# Gtfs2sumo

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![License](https://img.shields.io/badge/license-EPL--2.0-lightgrey.svg)

`Gtfs2sumo` is a Python utility library for converting [GTFS](https://gtfs.org/) transit feeds into [SUMO](https://www.eclipse.dev/sumo/) simulation-ready routes and stops. Built as an extension to [gtfs2pt.py](https://github.com/eclipse-sumo/sumo/blob/main/tools/import/gtfs/gtfs2pt.py), `gtfs2sumo` covers a use case not yet supported by [gtfs2pt.py](https://github.com/eclipse-sumo/sumo/blob/main/tools/import/gtfs/gtfs2pt.py).


</div>


## 🚀 Overview

- Converts routes and schedules while staying true to the shape of the routes.
- Requires at least the [GTFS](https://gtfs.org/) feed (including shapes), an existing [SUMO](https://www.eclipse.dev/sumo/) network, and a date (YYYYMMDD) for which to extract the schedule.
- Offers an option to generate maps for transparency and additional information with `verbose`.
- For more options, check out [`convert_gtfs2sumo.py`](convert_gtfs2sumo.py).
- Complementary to [gtfs2pt.py](https://github.com/eclipse-sumo/sumo/blob/main/tools/import/gtfs/gtfs2pt.py).
- It is highly recommended to reduce the [GTFS](https://gtfs.org/) feed to the wanted routes first before using `**`Gtfs2Sumo`. This helps to focus on relevant data and ensures more efficient processing.

## 🔧 Installation

We recommend using a virtual environment to manage dependencies.
Create and activate a virtual environment with Python version `3.11.10` 
```bash
    pip install -r requirements.txt
```


## 🧪 Example Usage

```bash
from convert_gtfs2sumo import GTFS2SUMO

gtfs =  GTFS2SUMO(
    net_file_path='path/to/your/network', 
    gtfs_data_path='path/to/your/gtfs/dataset', 
    gtfs_date="YYYYMMDD",
    create_maps=True, verbose=True
)
gtfs.extract_sumo_traffic()
```
For more options, check out [`convert_gtfs2sumo.py`](convert_gtfs2sumo.py).  
An example notebook can be found here: [`example.ipynb`](example.ipynb).

## 🙏 Acknowledgements

This library was developed as part of the master's thesis "Modelling Microscopic Urban Bus Traffic for the BeST Simulation Scenario." It serves to extend the BeST Scenario with Berlin bus traffic data. Special thanks to Moritz Schweppenhäuser for his supervision and contributions to the extensions of **gtfs2sumo**.


## 📚 References

If you use `gtfs2sumo` in your research, please provide a link to this repository, or by citing this reference:

> Schweppenhäuser M., Großmann T., Schrab K., Protzmann, R., Radusch, I. (2025). *Modeling Bus Traffic for the Berlin SUMO Traffic Scenario*. SUMO User Conference 2025. https://sumo.dlr.de/pdf/2025/pre-print-2613.pdf

## 📄 License

This repository includes and modifies [gtfs2pt.py](https://github.com/eclipse-sumo/sumo/blob/main/tools/import/gtfs/gtfs2pt.py) from the [Eclipse SUMO](https://www.eclipse.dev/sumo/) project, which is licensed under the [Eclipse Public License v2.0](https://www.eclipse.org/legal/epl-2.0/).

Please cite Eclipse SUMO appropriately if you use this tool in your research or development.