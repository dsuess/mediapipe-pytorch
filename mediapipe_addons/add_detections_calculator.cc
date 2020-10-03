#include <vector>
#include <iostream>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"

namespace mediapipe
{
  typedef std::vector<Detection> Detections;
  const char kDetections[] = "DETECTIONS";

  class AddDetectionCalculator : public CalculatorBase
  {
  public:
    AddDetectionCalculator() = default;
    ~AddDetectionCalculator() override = default;

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
      auto output_detections = absl::make_unique<Detections>();

      Detection detection;
      //detection.set_label_id(1);
      //detection.set_label("HELLO");
      auto location = detection.mutable_location_data();
      location->set_format(LocationData::BOUNDING_BOX);
      auto box = location->mutable_bounding_box();
      box->set_xmin(10);
      box->set_ymin(10);
      box->set_width(100);
      box->set_height(100);
      output_detections->emplace_back(detection);

      cc->Outputs()
          .Index(0)
          .Add(output_detections.release(), cc->InputTimestamp());

      return ::mediapipe::OkStatus();
    }
  };

  REGISTER_CALCULATOR(AddDetectionCalculator);
} // namespace mediapipe
