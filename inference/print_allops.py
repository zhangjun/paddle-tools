import paddle
import paddle
import paddle.static as static
import numpy as np
import json
import os
from paddle.fluid import core
from paddle.fluid import executor
from paddle.fluid.io import load_inference_model
from paddle.fluid.framework import IrGraph

paddle.enable_static()

class InferModel(object):
    def __init__(self, list):
        self.program = list[0]
        self.feed_var_names = list[1]
        self.fetch_vars = list[2]


def printAllOps(model_path, model, params):
    place = paddle.CPUPlace()
    # place = core.CUDAPlace(3)
    # scope = global_scope()
    # device = "gpu" if core.is_compiled_with_cuda() else "cpu"
    exe = paddle.static.Executor(place)
    [inference_program, feed_target_names,
     fetch_targets] = paddle.static.load_inference_model(
         path_prefix=model_path,
         executor=exe)
         # model_filename=model,
         # params_filename=params)
    graph = IrGraph(core.Graph(inference_program.desc), for_test=True)
    op_nodes = graph.all_op_nodes()
    op_lists = [] 
    for op_node in op_nodes:
        op_lists.append(op_node.name())
    ops = set(op_lists)
    print(model_path, sorted(ops))


model = 'inference'
params = 'inference'
model_path = '/mydev/work/test/infer_bench/ShuffleNetV2_x0_5/inference'
printAllOps(model_path, model, params)

