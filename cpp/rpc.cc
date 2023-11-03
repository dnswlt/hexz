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

absl::StatusOr<KeyedModel> RPCClient::FetchLatestModel() {
  // Get info about the current model.
  cpr::Response key_resp =
      cpr::Get(cpr::Url{training_server_url_ + "/models/current"},  //
               cpr::Timeout{1000},                                  //
               cpr::Header{{"Accept", "application/x-protobuf"}});
  if (key_resp.status_code == 0) {
    return absl::UnavailableError(
        absl::StrCat("Server unreachable: ", key_resp.error.message));
  }
  if (key_resp.status_code != 200) {
    return absl::AbortedError(
        absl::StrCat("Server returned status code ", key_resp.status_code));
  }
  hexzpb::ModelKey model_key;
  if (auto status = google::protobuf::util::JsonStringToMessage(
          key_resp.text, &model_key,
          google::protobuf::util::JsonParseOptions());
      !status.ok()) {
    return status;
  }
  ABSL_LOG(INFO) << "Fetching model " << model_key.name() << ":"
                 << model_key.checkpoint();
  cpr::Response model_resp = cpr::Get(
      cpr::Url{absl::StrCat(training_server_url_, "/models/", model_key.name(),
                            "/checkpoints/", model_key.checkpoint())},  //
      cpr::Parameters{{"repr", "scriptmodule"}},                        //
      cpr::Timeout{1000});
  if (model_resp.status_code == 404) {
    // This *might* be caused by a concurrent model change between our call to
    // /models/current and the failed fetch. We should try again.
    return absl::NotFoundError("Server returned 404");
  }
  if (model_resp.status_code != 200) {
    return absl::AbortedError(absl::StrCat(
        "Fetch model: server returned status code ", model_resp.status_code));
  }
  try {
    std::istringstream model_is(model_resp.text);
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
    const std::vector<hexzpb::TrainingExample>& examples) {
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
  if (resp.status_code != 200) {
    return absl::AbortedError(
        absl::StrCat("Server did not like our request: ", resp.status_code));
  }
  hexzpb::AddTrainingExamplesResponse response;
  if (!response.ParseFromString(resp.text)) {
    return absl::InternalError(
        "Cannot parse reponse as AddTrainingExamplesResponse");
  }
  return response;
}

}  // namespace hexz
