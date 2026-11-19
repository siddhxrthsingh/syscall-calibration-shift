import copy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
from syscalib.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def small_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["stream"].update({"n_pre": 3, "n_post": 4, "batch_size": 200, "n_init_cal": 200,
                        "eval_benign": 400, "eval_attack_per_family": 100, "oracle_ref": 1500})
    c["detector"]["n_train"] = 800
    return c
