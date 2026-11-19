"""Keyed random streams.

Every logically separate data set draws from its own stream, keyed by
(seed, stream_id, index). Two consequences:
  * training / calibration / evaluation / oracle data can never share random numbers;
  * pre-shift batches are identical across workload conditions (common random
    numbers), so stationary vs shifted comparisons are paired.
"""
import numpy as np

STREAMS = {
    "world": 0,
    "train": 1,
    "init_cal": 2,
    "cal_benign": 3,
    "cal_attack_base": 4,
    "cal_attack_mod": 5,
    "cal_family": 6,
    "cal_uniform": 7,
    "eval_benign": 8,
    "eval_attack_base": 9,
    "eval_attack_mod": 10,
    "oracle": 11,
    "attack_design": 12,
}


def stream(seed, name, index=0, sub=0):
    return np.random.default_rng([int(seed), STREAMS[name], int(index), int(sub)])
