#!/usr/bin/env python3
"""
ROS launch entrypoint that reuses YOPO/run/replay_gcopter_traj.py.
"""

import sys
from pathlib import Path


def _find_collect_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "YOPO" / "run" / "replay_gcopter_traj.py").exists():
            return p
    raise RuntimeError("Cannot locate collect repo root from script path.")


def main():
    here = Path(__file__).resolve()
    collect_root = _find_collect_root(here)
    yopo_run = collect_root / "YOPO" / "run"
    sys.path.insert(0, str(yopo_run))

    from replay_gcopter_traj import main as replay_main

    replay_main()


if __name__ == "__main__":
    main()

