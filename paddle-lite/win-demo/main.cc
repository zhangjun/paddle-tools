// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "opencv2/opencv.hpp"

using namespace paddle::lite_api;  // NOLINT

#if _WIN32
	#pragma comment(lib,"libpaddle_api_light_bundled.lib")
#endif


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

void normalize_image(
    const std::vector<float> &mean,
    const std::vector<float> &scale,
    cv::Mat& im, // NOLINT
    float* input_buffer) {
  int height = im.rows;
  int width = im.cols;
  int stride = width * height;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int base = h * width + w;
      input_buffer[base + 0 * stride] =
          (im.at<cv::Vec3f>(h, w)[0] - mean[0]) * scale[0];
      input_buffer[base + 1 * stride] =
          (im.at<cv::Vec3f>(h, w)[1] - mean[1]) * scale[1];
      input_buffer[base + 2 * stride] =
          (im.at<cv::Vec3f>(h, w)[2] - mean[2]) * scale[2];
    }
  }
}

void preprocess(const cv::Mat& image_mat, std::vector<float> input_data, float shrink = 0.6) {
  // Clone the image : keep the original mat for postprocess
  cv::Mat im = image_mat.clone();
  cv::resize(im, im, cv::Size(), shrink, shrink, cv::INTER_CUBIC);
  im.convertTo(im, CV_32FC3, 1.0);
  int rc = im.channels();
  int rh = im.rows;
  int rw = im.cols;
  //input_shape_ = {1, rc, rh, rw};
  input_data.resize(1 * rc * rh * rw);
  float* buffer = input_data.data();
  //normalize_image(mean_, scale_, im, input_data_.data());
}

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

  config.set_model_from_file(model_file);

  //std::string model_buffer = ReadFile(model_file); 
  //config.set_model_from_buffer(model_buffer);
  
  //config.set_power_mode(LITE_POWER_HIGH);
  //config.set_threads(1);

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);
  return predictor;
}

static void run_predict(std::shared_ptr<PaddlePredictor>& predictor, const std::vector<shape_t>& input_shapes, cv::Mat &img) {
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

	predictor->Run();
	
	std::unique_ptr<const paddle::lite_api::Tensor> output_tensor =
        predictor->GetOutput(0);
	
	auto out_shape = output_tensor->shape();
    auto out_data = output_tensor->data<float>();
    auto out_mean = compute_mean<float>(out_data, ShapeProduction(out_shape));
    auto out_std_dev = compute_standard_deviation<float>(
        out_data, ShapeProduction(out_shape), true, out_mean);
	std::cout << "output shape(NCHW):" << ShapePrint(out_shape) 
	          << ", std-dev:" << out_std_dev 
			  << ", mean:" << out_mean << std::endl;
}

static void handle_camera(std::shared_ptr<PaddlePredictor>& predictor0, std::shared_ptr<PaddlePredictor>& predictor1)
{
    cv::VideoCapture mVideoCapture(0);
	if(!mVideoCapture.isOpened()) {
		std::cout << "failed to open camera." << std::endl;
		return;
	}
	//mVideoCapture.set(3, 640);
	//mVideoCapture.set(4, 480);
	
    cv::Mat frame;
    mVideoCapture >> frame;
    while (!frame.empty()) {
        mVideoCapture >> frame;
        if (frame.empty()) {
            break;
        }
        auto start_time = std::chrono::steady_clock::now();
		std::cout << "run " << std::endl;
		// predict
		 std::vector<shape_t> shape0{{1, 4, 160, 320}};
		 std::vector<shape_t> shape1{{1, 4, 128, 256}};
		 run_predict(predictor0, shape0, frame);
		 run_predict(predictor1, shape1, frame);
		
		auto end_time = std::chrono::steady_clock::now();
        double diff = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << "cost: " << diff / 1000.0f << " ms" << std::endl;
        cv::imshow("img", frame);
        cv::waitKey(1);
    }
	cv::waitKey();
	frame.release();
    return;
}

class Predictor {
public:
	Predictor() {};
	bool init(){
		p0_ = CreatePredictor(model_file0_);
		p1_ = CreatePredictor(model_file1_);
		return true;
	}
	bool set_model(const std::string& model_file0, const std::string& model_file1) {
		model_file0_ = model_file0;
		model_file1_ = model_file1;
		return true;
	}
	bool handle() {
		handle_camera(p0_, p1_);
		return true;
	}
	
	bool release() {
		return true;
	}
	~Predictor() {}
private:
    std::shared_ptr<PaddlePredictor> p0_;
	std::shared_ptr<PaddlePredictor> p1_;
	std::string model_file0_;
	std::string model_file1_;
};

int main(int argc, char** argv){
	Predictor* predictor = new Predictor();
	predictor -> set_model("./model/model_v1.nb", "./model/model_v1_small.nb");
	predictor -> init();
	
    predictor -> handle();
	
	predictor -> release();
	if(predictor != nullptr) {
		delete predictor;
		predictor = nullptr;
	}
    return 0;
}