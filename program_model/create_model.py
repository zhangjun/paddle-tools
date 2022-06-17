import os
import sys
import shutil

import numpy as np
import paddle
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest
from auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
from program_config import TensorConfig, OpConfig, ProgramConfig, create_fake_model, create_quant_model

class NearestInterpV2():
    def __init__(self):
        np.random.seed(1024)
        paddle.enable_static()

    def generate_op_config(self,
                           ops_config: List[Dict[str, Any]]) -> List[OpConfig]:
        ops = []
        for i in range(len(ops_config)):
            op_config = ops_config[i]
            ops.append(
                OpConfig(
                    type=op_config['op_type'],
                    inputs=op_config['op_inputs'],
                    outputs=op_config['op_outputs'],
                    attrs=op_config['op_attrs']))
        return ops

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input():
            return np.ones([1, 18, 28, 28]).astype(np.float32)

        ops_config = [{
            "op_type": "nearest_interp",
            "op_inputs": {
                "X": ["input_data"]
            },
            "op_outputs": {
                "Out": ["interp_output_data"]
            },
            "op_attrs": {
                "data_layout": "NCHW",
                "interp_method": "nearest",
                "align_corners": True,
                "align_mode": 1,
                # "scale": [2., 2.],
                "scale": 2.,
                "out_d": 0,
                "out_h": 0,
                "out_w": 0
            # "op_attrs": {
            #     "data_layout": "NCHW",
            #     "interp_method": "nearest",
            #     "align_corners": False,
            #     "align_mode": 1,
            #     "scale": [2., 2.],
            #     "out_d": 0,
            #     "out_h": 0,
            #     "out_w": 0
            }
        }]

        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={"input_data": TensorConfig(data_gen=generate_input)},
            outputs=["interp_output_data"])

        # yield program_config
        return program_config

interp_op = NearestInterpV2() 
prog_config = interp_op.sample_program_configs()
print(prog_config)
# sys.exit(0)
model, params = create_fake_model(prog_config)
model_path = "near_interp"
if os.path.exists(model_path):
    shutil.rmtree(model_path)
if not os.path.exists(model_path):
    os.mkdir(model_path)
with open(model_path + "/inference.pdmodel", "wb") as f:
    f.write(model)
with open(model_path + "/inference.pdiparams", "wb") as f:
    f.write(params)
