// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "paddle_api.h"  // NOLINT
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library:libpaddle_api_light_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
// #include "paddle_use_kernels.h"  // NOLINT
// #include "paddle_use_ops.h"      // NOLINT

using namespace paddle::lite_api;  // NOLINT

std::string ReadFile(const std::string& filename) {
  std::ifstream ifile(filename.c_str());
  if (!ifile.is_open()) {
    std::cout << "Open file: [" << filename << "] failed.";
  }
  std::ostringstream buf;
  char ch;
  while (buf && ifile.get(ch)) buf.put(ch);
  ifile.close();
  return buf.str();
}

class Timer {
 private:
  std::chrono::high_resolution_clock::time_point inTime, outTime;

 public:
  void startTimer() { inTime = std::chrono::high_resolution_clock::now(); }

  // unit millisecond
  float getCostTimer() {
    outTime = std::chrono::high_resolution_clock::now();
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(outTime - inTime)
            .count() /
        1e+3);
  }
};

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

std::string ShapePrint(const std::vector<shape_t>& shapes) {
  std::string shapes_str{""};
  for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx) {
    auto shape = shapes[shape_idx];
    std::string shape_str;
    for (auto i : shape) {
      shape_str += std::to_string(i) + ",";
    }
    shapes_str += shape_str;
    shapes_str +=
        (shape_idx != 0 && shape_idx == shapes.size() - 1) ? "" : " : ";
  }
  return shapes_str;
}

std::string ShapePrint(const shape_t& shape) {
  std::string shape_str{""};
  for (auto i : shape) {
    shape_str += std::to_string(i) + " ";
  }
  return shape_str;
}

std::vector<std::string> split_string(const std::string& str_in) {
  std::vector<std::string> str_out;
  std::string tmp_str = str_in;
  while (!tmp_str.empty()) {
    size_t next_offset = tmp_str.find(":");
    str_out.push_back(tmp_str.substr(0, next_offset));
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp_str = tmp_str.substr(next_offset + 1);
    }
  }
  return str_out;
}

std::vector<int64_t> get_shape(const std::string& str_shape) {
  std::vector<int64_t> shape;
  std::string tmp_str = str_shape;
  while (!tmp_str.empty()) {
    int dim = atoi(tmp_str.data());
    shape.push_back(dim);
    size_t next_offset = tmp_str.find(",");
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp_str = tmp_str.substr(next_offset + 1);
    }
  }
  return shape;
}

template <typename T>
double compute_mean(const T* in, const size_t length) {
  double sum = 0.;
  for (size_t i = 0; i < length; ++i) {
    sum += in[i];
  }
  return sum / length;
}

template <typename T>
double compute_standard_deviation(const T* in,
                                  const size_t length,
                                  bool has_mean = false,
                                  double mean = 10000) {
  if (!has_mean) {
    mean = compute_mean<T>(in, length);
  }

  double variance = 0.;
  for (size_t i = 0; i < length; ++i) {
    variance += pow((in[i] - mean), 2);
  }
  variance /= length;
  return sqrt(variance);
}

std::shared_ptr<PaddlePredictor> CreatePredictor(std::string model_file) {
  // 1. Set MobileConfig
  MobileConfig config;

  //config.set_model_from_file(model_file);

  std::string model_buffer = ReadFile(model_file); 
  config.set_model_from_buffer(model_buffer);
  
  config.set_power_mode(LITE_POWER_HIGH);
  config.set_threads(1);


  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);
  return predictor;
}

void RunPredictor(std::shared_ptr<PaddlePredictor> predictor, const std::vector<shape_t>& input_shapes) {
  // 3. Prepare input data
  for (int j = 0; j < input_shapes.size(); ++j) {
    auto input_tensor = predictor->GetInput(j);
    input_tensor->Resize(input_shapes[j]);
    auto input_data = input_tensor->mutable_data<float>();
    int input_num = 1;
    for (int i = 0; i < input_shapes[j].size(); ++i) {
      input_num *= input_shapes[j][i];
    }

    for (int i = 0; i < input_num; ++i) {
      input_data[i] = 1.f;
    }
  }

  // 4. Run predictor
  auto begin = std::chrono::steady_clock::now();
  predictor->Run();
  auto end = std::chrono::steady_clock::now();
  long long cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

  // 5. Get output
  size_t output_tensor_num = predictor->GetOutputNames().size();

  for (size_t tidx = 0; tidx < output_tensor_num; ++tidx) {
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor =
        predictor->GetOutput(tidx);
    auto out_shape = output_tensor->shape();
    auto out_data = output_tensor->data<float>();
    auto out_mean = compute_mean<float>(out_data, ShapeProduction(out_shape));
    auto out_std_dev = compute_standard_deviation<float>(
        out_data, ShapeProduction(out_shape), true, out_mean);

    std::cout << "[0-" << tidx
              << "] std-dev:" << out_std_dev << ", mean:" << out_mean
	      << ", cost:" << cost << " ms." << std::endl;

  }
}

int main(int argc, char** argv) {
  std::string model_0 = "/mydev/zhangjun/data/infoflow_model/model_v1.nb";
  std::string model_1 = "/mydev/zhangjun/data/infoflow_model/model_v1_small.nb";
  std::vector<shape_t> input_shapes0{{1, 4, 160, 320}};  // shape_t ==> std::vector<int64_t>
  std::vector<shape_t> input_shapes1{{1, 4, 128, 256}};  // shape_t ==> std::vector<int64_t>


  auto p0 = CreatePredictor(model_0);
  auto p1 = CreatePredictor(model_1);

  // warm up
  std::cout << "warmup begin." << std::endl;
  for(int i = 0; i < 10; ++ i) {
    RunPredictor(p0, input_shapes0);
    RunPredictor(p1, input_shapes1);
  }
  std::cout << "warmup end." << std::endl;

  // run model
  while(1) {
    RunPredictor(p0, input_shapes0);
    RunPredictor(p1, input_shapes1);
  }


  return 0;
}
