#include <vector>
#include <iostream>

#include "torch/script.h"
#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"

namespace mediapipe
{
  using namespace torch::indexing;

  std::unique_ptr<torch::Tensor> ImageFrameToNormalizedTensor(
      const mediapipe::ImageFrame &image_frame, float mean, float stddev)
  {
    const int cols = image_frame.Width();
    const int rows = image_frame.Height();
    const int channels = image_frame.NumberOfChannels();
    const uint8 *pixel = image_frame.PixelData();
    const int width_padding = image_frame.WidthStep() - cols * channels;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    auto tensor_ = torch::empty({channels, rows, cols}, options);
    auto tensor = absl::make_unique<torch::Tensor>(tensor_);
    auto tensor_data = tensor->accessor<float, 3>();

    for (int row = 0; row < rows; ++row)
    {
      for (int col = 0; col < cols; ++col)
      {
        for (int channel = 0; channel < channels; ++channel)
        {
          tensor_data[channel][row][col] = (float(pixel[channel]) - mean) / stddev;
        }
        pixel += channels;
      }
      pixel += width_padding;
    }
    return tensor;
  }

  typedef std::vector<Detection> Detections;
  const char kDetections[] = "DETECTIONS";

  class PytorchInferenceCalculator : public CalculatorBase
  {
  private:
    torch::jit::script::Module model;

  public:
    PytorchInferenceCalculator()
    {
      // FIXME Move this into side-channel
      std::cout << "Loading da model" << std::endl;
      model = torch::jit::load("examples/yolov5s.torchscript.pt");
      model.eval();
    };
    ~PytorchInferenceCalculator() override = default;

    static ::mediapipe::Status GetContract(CalculatorContract *cc)
    {
      // We need to add a single input to get timing right
      cc->Inputs().Index(0).Set<ImageFrame>();
      cc->Outputs().Index(0).Set<Detections>();
      return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status Open(CalculatorContext *cc) override
    {
      cc->SetOffset(TimestampDiff(0));
      return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status Process(CalculatorContext *cc) override
    {
      const ImageFrame &frame = cc->Inputs().Index(0).Value().Get<ImageFrame>();
      auto tensor = ImageFrameToNormalizedTensor(frame, 0, 255.);
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(tensor->unsqueeze(0));
      auto result = model.forward(inputs).toTensor();

      // Post processing
      result = result.squeeze(0);
      auto confidence = result.index({Slice(), 4});
      result = result.index({confidence > 0.5, Slice()});

      auto output_detections = absl::make_unique<Detections>();
      for (auto idx = 0; idx < result.size(0); ++idx)
      {
        const float x = result.index({idx, 0}).item<float>();
        const float y = result.index({idx, 1}).item<float>();
        const float w = result.index({idx, 2}).item<float>();
        const float h = result.index({idx, 3}).item<float>();
        const float obj_score = result.index({idx, 4}).item<float>();
        const auto class_scores = result.index({idx, Slice(5, None)});
        const auto class_id = class_scores.argmax().item<int>();
        const auto class_score = class_scores[class_id].item<float>();

        Detection detection;
        detection.add_label_id(class_id);
        detection.add_score(obj_score * class_score);
        //detection.set_label("HELLO");
        auto location = detection.mutable_location_data();
        location->set_format(LocationData::RELATIVE_BOUNDING_BOX);
        auto box = location->mutable_relative_bounding_box();
        box->set_xmin((x - w / 2) / 640);
        box->set_ymin((y - h / 2) / 480);
        box->set_width(w / 640);
        box->set_height(h / 480);
        output_detections->emplace_back(detection);
      }

      cc->Outputs()
          .Index(0)
          .Add(output_detections.release(), cc->InputTimestamp());

      return ::mediapipe::OkStatus();
    }
  };

  REGISTER_CALCULATOR(PytorchInferenceCalculator);
} // namespace mediapipe
