#include "rpc.h"

#include <absl/status/statusor.h>
#include <cpr/cpr.h>
#include <google/protobuf/util/json_util.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <vector>

#include "base.h"
#include "hexz.pb.h"

namespace hexz {

using google::protobuf::RepeatedPtrFieldBackInserter;

absl::StatusOr<KeyedModel> RPCClient::FetchLatestModel() {
  // Download the latest model.
  cpr::Response resp =
      cpr::Get(cpr::Url{training_server_url_ + "/models/latest"},   //
               cpr::Timeout{1000},                                  //
               cpr::Parameters{{"repr", "scriptmodule"}},           //
               cpr::Header{{"Accept", "application/octet-stream"}}  //
      );
  if (resp.status_code == 0) {
    return absl::UnavailableError(
        absl::StrCat("Server unreachable: ", resp.error.message));
  }
  if (resp.status_code != 200) {
    return absl::AbortedError(
        absl::StrCat("Server returned status code ", resp.status_code));
  }
  if (resp.header["X-Model-Key"] == "") {
    return absl::InternalError(
        "Server did not respond with X-Model-Key header");
  }
  hexzpb::ModelKey model_key;
  if (auto status = google::protobuf::util::JsonStringToMessage(
          resp.header["X-Model-Key"], &model_key,
          google::protobuf::util::JsonParseOptions());
      !status.ok()) {
    return status;
  }
  ABSL_LOG(INFO) << "Fetched model " << model_key.name() << ":"
                 << model_key.checkpoint();
  try {
    std::istringstream model_is(resp.text);
    auto model = torch::jit::load(model_is);  // , torch::kCPU);
    model.to(torch::kCPU);
    model.eval();
    return KeyedModel{
        .key = model_key,
        .model = model,
    };
  } catch (const c10::Error& e) {
    return absl::InternalError(
        absl::StrCat("Failed to load torch module: ", e.msg()));
  }
}

absl::StatusOr<hexzpb::AddTrainingExamplesResponse> RPCClient::SendExamples(
    const hexzpb::ModelKey& key,
    std::vector<hexzpb::TrainingExample>&& examples) {
  hexzpb::AddTrainingExamplesRequest req;
  *req.mutable_model_key() = key;
  req.mutable_examples()->Reserve(examples.size());
  std::move(examples.begin(), examples.end(),
            RepeatedPtrFieldBackInserter(req.mutable_examples()));
  cpr::Response resp =
      cpr::Post(cpr::Url{training_server_url_ + "/examples"},             //
                cpr::Timeout{1000},                                       //
                cpr::Header{{"Content-Type", "application/x-protobuf"}},  //
                cpr::Body{req.SerializeAsString()});
  if (resp.status_code == 0) {
    return absl::UnavailableError(
        absl::StrCat("Server unreachable: ", resp.error.message));
  }
  if (resp.status_code != 200) {
    return absl::AbortedError(
        absl::StrCat("Server responded did not like our request: ", resp.status_code));
  }
  hexzpb::AddTrainingExamplesResponse response;
  if (!response.ParseFromString(resp.text)) {
    return absl::InternalError(
        "Cannot parse reponse as AddTrainingExamplesResponse");
  }
  return response;
}

}  // namespace hexz
