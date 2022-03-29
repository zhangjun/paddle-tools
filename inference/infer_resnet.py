import numpy as np
import argparse
import cv2

from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType


def init_predictor(args):
    if args.model_dir is not "":
        config = Config(args.model_dir)
    else:
        config = Config(args.model_file, args.params_file)

    config.enable_memory_optim()
    # config.enable_profile()
    if args.use_gpu:
        config.enable_use_gpu(1000, 0)
        if args.use_trt:
            if args.precision == 'fp32':
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    max_batch_size=args.batch_size,
                    min_subgraph_size=5,
                    precision_mode=PrecisionType.Float32,
                    use_static=False,
                    use_calib_mode=False)
            elif args.precision == 'fp16':
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    max_batch_size=args.batch_size,
                    min_subgraph_size=5,
                    precision_mode=PrecisionType.Half,
                    use_static=False,
                    use_calib_mode=False)
    else:
        # If not specific mkldnn, you can set the blas thread.
        # The thread num should not be greater than the number of cores in the CPU.
        config.set_cpu_math_library_num_threads(4)
        config.enable_mkldnn()

    predictor = create_predictor(config)
    return predictor


def run(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())

    # do the inference
    for i in range(2):
        predictor.run()

    begin = time.time()
    for i in range(10):
        predictor.run()
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
    end = time.time()
    print("cost: ", (end - begin) * 1000 / 10)

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="Model filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="",
        help=
        "Parameter filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help=
        "Model dir, If you load a non-combined model, specify the directory of the model."
    )
    parser.add_argument("--use_gpu",
                        type=int,
                        default=0,
                        help="Whether use gpu.")
    parser.add_argument("--use_trt",
                        type=int,
                        default=0,
                        help="Whether use trt.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="batch size.")
    parser.add_argument('--precision',
                        type=str,
                        default='fp32',
                        choices=["fp32", "fp16"],
                        help="precision mode.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pred = init_predictor(args)
    #img = cv2.imread('./ILSVRC2012_val_00000247.jpeg')
    #img = preprocess(img)
    #img = np.ones((1, 3, 224, 224)).astype(np.float32)
    img = np.ones((1, 3, 224, 224)).astype(np.float32)
    result = run(pred, [img])
    # img0 = np.ones((1, 2)).astype(np.float32)
    # img1 = np.ones((1, 3, 160, 160)).astype(np.float32)
    # img2 = np.ones((1, 2)).astype(np.float32)
    # result = run(pred, [img0, img1, img2])
    print("std: ", np.std(result[0]))
    print("mean: ", np.mean(result[0]))
