import json
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"
EVAL_STATES_DIR = PACKAGE_DIR / "eval_states"

# the tasks import ASSETS_DIR from this module, so they are imported after it
from env.cube import Cube
from env.drawer import Drawer
from env.puzzle import Puzzle
from env.tool import Tool

ENVS = {"cube": Cube, "drawer": Drawer, "puzzle": Puzzle, "tool": Tool}


def load_eval_states(env_name: str) -> dict:
    """
    Start and goal states of the 30 evaluation episodes of a task, e.g. `d['ep3']['goal_0']['state']`.
    A goal entry also has `render_start`, the start index to pass to `render_state` with that goal state
    (in the drawer task, the pre-drawn strokes depend on the start).
    """
    def as_arrays(x):
        return {k: as_arrays(v) for k, v in x.items()} if isinstance(x, dict) else np.asarray(x, dtype=np.float32)

    with open(EVAL_STATES_DIR / f"{env_name}.json") as f:
        episodes = json.load(f)
    for episode in episodes.values():
        for entry in episode.values():
            entry["state"] = as_arrays(entry["state"])
    return episodes
