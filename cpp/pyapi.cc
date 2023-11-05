#include "pyapi.h"

#include <absl/log/absl_log.h>
#include <absl/strings/str_format.h>
#include <torch/script.h>

#include <fstream>
#include <memory>
#include <stdexcept>

#include "hexz.pb.h"

namespace hexz {

class MoveSuggester::impl {
 public:
  std::string SuggestMove(const std::string& request);
  void LoadModel(const std::string& path);

 private:
  torch::jit::Module module_;
};

// Must be in the .cc file to make pimpl with std::unique_ptr work.
MoveSuggester::MoveSuggester() : pimpl_{std::make_unique<impl>()} {}
MoveSuggester::~MoveSuggester() = default;
MoveSuggester::MoveSuggester(MoveSuggester&&) = default;
MoveSuggester& MoveSuggester::operator=(MoveSuggester&& other) = default;

std::string MoveSuggester::SuggestMove(const std::string& request) {
  return pimpl()->SuggestMove(request);
}

void MoveSuggester::LoadModel(const std::string& path) {
  return pimpl()->LoadModel(path);
}

std::string MoveSuggester::impl::SuggestMove(const std::string& request) {
  ABSL_DLOG(INFO) << "SuggestMove: request of length " << request.size();
  hexzpb::SuggestMoveRequest req;
  if (!req.ParseFromString(request)) {
    throw std::invalid_argument("not a valid SuggestMoveRequest proto");
  }

  hexzpb::SuggestMoveResponse resp;
  auto& move = *resp.mutable_move();
  move.set_move(0);
  move.set_cell_type(hexzpb::Field::FLAG);
  move.set_row(0);
  move.set_col(0);
  move.set_player_num(1);
  return resp.SerializeAsString();
}

void MoveSuggester::impl::LoadModel(const std::string& path) {
  ABSL_LOG(INFO) << "Loading model from " << path;
  try {
    std::ifstream f_in(path, std::ios::binary);
    if (!f_in.is_open()) {
      throw new std::invalid_argument("cannot read model from path " + path);
    }
    auto m = torch::jit::load(f_in);
    m.to(torch::kCPU);
    m.eval();
    module_ = m;
    ABSL_LOG(INFO)
        << "Model loaded successfully. Ready to serve SuggestMove requests!";

  } catch (const c10::Error& e) {
    throw new std::runtime_error("torch::jit::load(" + path + "): " + e.msg());
  }
}

}  // namespace hexz
