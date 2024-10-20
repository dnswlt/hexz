#include "mcts.h"

#include <absl/log/absl_log.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>
#include <torch/torch.h>

#include <algorithm>
#include <boost/fiber/all.hpp>
#include <cassert>
#include <cmath>
#include <ostream>
#include <sstream>
#include <vector>

#include "base.h"
#include "perfm.h"

namespace hexz {

float Node::uct_c = 1.5;
float Node::initial_root_q_value = 0;
float Node::initial_q_penalty = 0;

void InitializeFromConfig(const Config& config) {
  Node::InitializeStaticMembers(config);
}

Node::Node(Node* parent, int turn, Move move)
    : parent_{parent}, turn_{turn}, move_{move} {}

float Node::Q() const noexcept {
  ABSL_DCHECK(!IsRoot())
      << "Must not call Q() on root node, it has no meaning.";
  if (visit_count_ == 0) {
    if (parent_->IsRoot()) {
      return initial_root_q_value;
    }
    float parentQ =
        parent_->NextTurn() != NextTurn() ? -parent_->Q() : parent_->Q();
    return std::max(-1.0f, parentQ - initial_q_penalty);
  }
  return wins_ / visit_count_;
}

int Node::NumVisitedChildren() const noexcept {
  int n = 0;
  for (const auto& c : children_) {
    if (c->visit_count_ > 0) {
      n++;
    }
  }
  return n;
}

namespace {
inline std::pair<int, int> MinMaxChildVisitCount(const Node& n) {
  int min_child_vc = 0;
  int max_child_vc = 0;
  if (!n.IsLeaf()) {
    const auto [min_c, max_c] =
        std::minmax_element(n.children().begin(), n.children().end(),
                            [](const auto& lhs, const auto& rhs) {
                              return lhs->visit_count() < rhs->visit_count();
                            });
    min_child_vc = (*min_c)->visit_count();
    max_child_vc = (*max_c)->visit_count();
  }
  return std::make_pair(min_child_vc, max_child_vc);
}
}  // namespace

std::string Node::Stats() const {
  const auto [min_vc, max_vc] = MinMaxChildVisitCount(*this);
  return absl::StrCat("nchildren:", children_.size(),              //
                      " visited_children:", NumVisitedChildren(),  //
                      " min_child_vc:", min_vc,                    //
                      " max_child_vc:", max_vc,                    //
                      " visit_count:", visit_count());
}

void Node::PopulateStats(hexzpb::TrainingExample::Stats& stats) const {
  const auto [min_vc, max_vc] = MinMaxChildVisitCount(*this);
  // Traverse tree to compute summary stats.
  int branch_nodes = 0;
  int max_depth = -1;
  int tree_size = 0;
  std::vector<std::pair<const Node*, int>> q;
  q.emplace_back(this, 0);
  std::vector<int32_t> nodes_per_depth;
  while (!q.empty()) {
    const auto [n, d] = q.back();
    q.pop_back();
    tree_size++;
    if (d > max_depth) {
      max_depth = d;
      nodes_per_depth.resize(max_depth + 1);
    }
    if (n->IsLeaf()) {
      continue;
    }
    nodes_per_depth[d]++;
    branch_nodes++;
    for (const auto& c : n->children()) {
      q.emplace_back(c.get(), d + 1);
    }
  }
  stats.set_valid_moves(children().size());
  stats.set_visit_count(visit_count());
  stats.set_visited_children(NumVisitedChildren());
  stats.set_search_depth(max_depth);
  stats.set_search_tree_size(tree_size);
  stats.set_branch_nodes(branch_nodes);
  stats.set_min_child_vc(min_vc);
  stats.set_max_child_vc(max_vc);
  stats.mutable_nodes_per_depth()->Assign(nodes_per_depth.begin(),
                                          nodes_per_depth.end());
}

float Node::Puct() const noexcept {
  ABSL_DCHECK(!IsRoot()) << "Prior: must not be called on root node";
  return Q() + Node::uct_c * prior_ * std::sqrt(parent_->visit_count_) /
                   (1 + visit_count_);
}

Node* Node::MaxPuctChild() const {
  //   Perfm::Scope ps(Perfm::MaxPuctChild);
  ABSL_CHECK(!children_.empty());
  Node* best = nullptr;
  float best_puct = std::numeric_limits<float>::lowest();
  for (const auto& c : children_) {
    float puct = c->Puct();
    if (puct > best_puct) {
      best = c.get();
      best_puct = puct;
    }
  }
  return best;
}

void Node::SetMoveProbs(torch::Tensor move_probs) {
  ABSL_DCHECK(move_probs.numel() == 2 * 11 * 10);
  ABSL_DCHECK(std::abs(move_probs.sum().item<float>() - 1.0) < 1e-3);
  const auto& move_probs_acc = move_probs.accessor<float, 3>();
  for (auto& c : children_) {
    const auto& m = c->move_;
    c->prior_ = move_probs_acc[static_cast<size_t>(m.typ)][m.r][m.c];
  }
}

torch::Tensor Node::Priors() const {
  auto t = torch::zeros({2, 11, 10});
  auto t_acc = t.accessor<float, 3>();
  for (const auto& c : children_) {
    const auto& m = c->move_;
    t_acc[static_cast<size_t>(m.typ)][m.r][m.c] = c->prior();
  }
  return t;
}

torch::Tensor Node::ActionMask() const {
  auto action_mask =
      torch::zeros({2, 11, 10}, c10::TensorOptions().dtype(torch::kBool));
  auto action_mask_acc = action_mask.accessor<bool, 3>();
  for (const auto& c : children()) {
    const auto& m = c->move_;
    action_mask_acc[static_cast<size_t>(m.typ)][m.r][m.c] = true;
  }
  return action_mask;
}

void Node::ShuffleChildren() {
  std::shuffle(children_.begin(), children_.end(), internal::rng);
}

void Node::AddDirichletNoise(float weight, float concentration) {
  std::vector<float> diri =
      internal::Dirichlet(children_.size(), concentration);
  int i = 0;
  for (auto& c : children_) {
    c->prior_ = (1 - weight) * c->prior_ + weight * diri[i];
    i++;
  }
}

torch::Tensor Node::NormVisitCounts() const {
  auto t = torch::zeros({2, 11, 10});
  auto t_acc = t.accessor<float, 3>();
  for (const auto& c : children_) {
    const auto& m = c->move_;
    t_acc[static_cast<size_t>(m.typ)][m.r][m.c] =
        static_cast<float>(c->visit_count_);
  }
  t /= t.sum();
  return t;
}

void Node::ResetTree() {
  wins_ = 0;
  visit_count_ = 0;
  for (auto& c : children_) {
    c->ResetTree();
  }
}

int Node::SampleChild() const {
  ABSL_CHECK(!children_.empty());
  float r = internal::UnitRandom();
  float s = 0;
  // The sum of the children's visit counts should be equal
  // to visit_count_ - 1. But summing explicitly (hopefully)
  // makes this more robust and future proof.
  int sum_vc = 0;
  for (const auto& c : children()) {
    sum_vc += c->visit_count_;
  }
  float vc = static_cast<float>(sum_vc);
  int i;
  for (i = 0; i < children_.size(); i++) {
    s += static_cast<float>(children_[i]->visit_count_) / vc;
    if (s >= r) {
      break;
    }
  }
  // Due to floating-point rounding issues, we *might* end up here with i ==
  // children_.size() if r ~= 1.0.
  if (i == children_.size()) {
    i--;
  }
  return i;
}

std::unique_ptr<Node> Node::SelectChildAsRoot(int i) {
  ABSL_CHECK(i >= 0 && i < children_.size());
  std::unique_ptr<Node> selected_child = std::move(children_[i]);
  selected_child->parent_ = nullptr;
  return selected_child;
}

inline int Node::MoveTurn() const noexcept {
  ABSL_DCHECK(!IsRoot());
  return parent_->turn_;
}

void Node::Backpropagate(float result) {
  Node* n = this;
  while (!n->IsRoot()) {
    n->visit_count_++;
    // wins_ aggregates the results from the perspective of the player
    // who makes the move_, not the player whose turn it is after
    // the move was made!
    n->wins_ += n->MoveTurn() == 0 ? result : -result;
    n = n->parent_;
  }
  // Update visit count of root node.
  n->visit_count_++;
}

void Node::CreateChildren(const std::vector<Move>& moves) {
  ABSL_DCHECK(children_.empty())
      << "Must not add children if they already exist.";
  // Initially we assume that it's the opponent's turn to make the next move.
  // If that turns out to be false, the turn gets updated when trying to find
  // next moves.
  int next_turn = 1 - NextTurn();
  children_.reserve(moves.size());
  for (int i = 0; i < moves.size(); i++) {
    children_.emplace_back(std::make_unique<Node>(this, next_turn, moves[i]));
  }
}

void Node::AppendDebugString(std::ostream& os,
                             const std::string& indent) const {
  os << indent << "Node(\n";
  os << indent << "  next_turn: " << NextTurn() << "\n";
  os << indent << "  move: " << move().DebugString() << "\n";
  os << indent << "  wins: " << wins() << "\n";
  os << indent << "  value: " << value() << "\n";
  os << indent << "  visit_count: " << visit_count() << "\n";
  if (!IsRoot()) {
    os << indent << "  puct: " << Puct() << "\n";
  }
  os << indent << "  nchildren: " << children().size() << "\n";
  if (!children().empty()) {
    os << indent << "  children: [\n";
    for (const auto& c : children()) {
      if (c->visit_count() > 0) {
        c->AppendDebugString(os, indent + "    ");
        os << indent << ",\n";
      }
    }
    os << indent << "  ]\n";
  }
  os << indent << ")\n";
}

std::string Node::DebugString() const {
  std::ostringstream os;
  AppendDebugString(os, "");
  return os.str();
}

namespace {
absl::Status WriteDotNodeRec(const Node& n, const std::string& name,
                             std::ofstream& os) {
  float puct = n.IsRoot() ? 0.0 : n.Puct() * 100;
  float q = n.IsRoot() ? 0.0 : n.Q();
  std::string label = absl::StrFormat(
      "%d(%d, %d, %d)%.0f v:%.2f\\nvc:%d q:%.2f p:%.1f c:%d",              //
      n.NextTurn(), n.move().typ, n.move().r, n.move().c, n.move().value,  //
      n.value(), n.visit_count(), q, puct, n.children().size());
  os << name << "[label=\"" << label << "\"]\n";
  int i = 0;
  for (const auto& c : n.children()) {
    if (c->visit_count() == 0) {
      continue;
    }
    auto child_name = name + "_" + std::to_string(i);
    auto status = WriteDotNodeRec(*c, child_name, os);
    if (!status.ok()) {
      return status;
    }
    os << name << " -> " << child_name << "\n";
    i++;
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status WriteDotGraph(const Node& root, const std::string& path) {
  std::ofstream os(path, std::ios::trunc);
  if (!os.is_open()) {
    return absl::AbortedError("Cannot open file " + std::string(path));
  }

  os << "digraph {\n";
  os << "node[shape=box]\n";
  std::string root_name = "n_0";
  auto status = WriteDotNodeRec(root, root_name, os);
  if (!status.ok()) {
    return status;
  }
  os << "}\n";
  return absl::OkStatus();
}

void TorchModel::SetDevice(torch::DeviceType device) {
  device_ = device;
  module_.to(device_);
}
void TorchModel::UpdateModel(hexzpb::ModelKey key,
                             torch::jit::Module&& module) {
  key_ = std::move(key);
  module_ = std::move(module);
  module_.to(device_);
}

TorchModel::Prediction TorchModel::Predict(const Board& board,
                                           const Node& node) {
  ABSL_CHECK(!node.IsLeaf()) << "Must not call Predict on a leaf node.";
  Perfm::Scope ps(Perfm::Predict);
  torch::NoGradGuard no_grad;
  auto board_input = board.Tensor(node.NextTurn()).unsqueeze(0).to(device_);
  auto action_mask = node.ActionMask().flatten().unsqueeze(0).to(device_);
  std::vector<torch::jit::IValue> inputs{
      board_input,
      action_mask,
  };
  auto output = module_.forward(inputs);

  // The model should output two values: the move likelihoods as a [1, 220]
  // tensor of logits, and a single float value prediction.
  ABSL_DCHECK(output.isTuple());
  const auto output_tuple = output.toTuple();
  ABSL_DCHECK(output_tuple->size() == 2);
  // Read logits back to CPU / main memory.
  const auto logits = output_tuple->elements()[0].toTensor().to(torch::kCPU);
  const auto dim = logits.sizes();
  ABSL_DCHECK(dim.size() == 2 && dim[0] == 1 && dim[1] == 2 * 11 * 10);
  const auto value = output_tuple->elements()[1].toTensor().item<float>();
  return Prediction{
      .move_probs = logits.reshape({2, 11, 10}).exp(),
      .value = value,
  };
}

std::vector<Model::Prediction> BatchedTorchModel::ComputeT::ComputeAll(
    std::vector<ComputeT::input_t>&& inputs) {
  Perfm::Scope ps(Perfm::PredictBatch);
  const size_t n_inputs = inputs.size();
  std::vector<torch::Tensor> boards;
  boards.reserve(n_inputs);
  std::vector<torch::Tensor> action_masks;
  action_masks.reserve(n_inputs);
  for (auto& inp : inputs) {
    boards.emplace_back(std::move(inp.board));
    action_masks.emplace_back(std::move(inp.action_mask));
  }
  std::vector<torch::jit::IValue> model_inputs = {
      torch::stack(boards).to(device_),
      torch::stack(action_masks).to(device_),
  };
  auto pred = module_.forward(model_inputs);
  ABSL_DCHECK(pred.isTuple());
  const auto output_tuple = pred.toTuple();
  ABSL_DCHECK(output_tuple->size() == 2);
  // Read logits back to CPU / main memory.
  const auto probs =
      output_tuple->elements()[0].toTensor().exp().to(torch::kCPU);
  const auto dim = probs.sizes();
  ABSL_DCHECK(dim.size() == 2 && dim[0] == n_inputs && dim[1] == 2 * 11 * 10);
  const auto values = output_tuple->elements()[1].toTensor().to(torch::kCPU);
  std::vector<Model::Prediction> result;
  result.reserve(n_inputs);
  for (int i = 0; i < n_inputs; i++) {
    result.push_back(Model::Prediction{
        .move_probs = probs.index({i}).reshape({2, 11, 10}),
        .value = values.index({i}).item<float>(),
    });
  }
  return result;
}

BatchedTorchModel::Prediction BatchedTorchModel::Predict(const Board& board,
                                                         const Node& node) {
  ABSL_CHECK(!node.IsLeaf()) << "Must not call Predict on a leaf node.";
  Perfm::Scope ps(Perfm::Predict);
  return batcher_.ComputeValue(ComputeT::input_t{
      .board = board.Tensor(node.NextTurn()),
      .action_mask = node.ActionMask().flatten(),
  });
}

void BatchedTorchModel::UpdateModel(hexzpb::ModelKey key,
                                    torch::jit::Module&& module) {
  key_ = std::move(key);
  batcher_.UpdateComputeT(std::make_unique<BatchedTorchModel::ComputeT>(
      std::move(module), device_));
}

hexzpb::ModelKey BatchedTorchModel::Key() const { return key_; }

inline void PlayoutRunner::Stats::Add(float result) noexcept {
  result_sum += result;
  runs++;
}

FiberTorchModel::FiberTorchModel(hexzpb::ModelKey key,
                                 torch::jit::Module module,
                                 torch::DeviceType device, int batch_size)
    : key_{std::move(key)},
      module_{std::move(module)},
      device_{device},
      batch_size_{batch_size} {
  // Acquire lock to have a synchronization point for module_ before any access
  // to it by other threads or fibers. (Necessary?)
  std::scoped_lock<std::mutex> lk(module_mut_);
  module_.to(device_);
  // Start the GPU thread only after this object has been
  // fully initialized.
  gpu_pipeline_thread_ = std::thread{&FiberTorchModel::RunGPUPipeline, this};
}

FiberTorchModel::~FiberTorchModel() {
  {
    std::scoped_lock<std::mutex> lk(request_mut_);
    ABSL_CHECK(active_fibers_ == 0)
        << "FiberTorchModel is being destroyed while fibers are active";
  }
  if (gpu_pipeline_thread_.joinable()) {
    gpu_pipeline_thread_.join();
  }
}

void FiberTorchModel::UpdateModel(hexzpb::ModelKey key,
                                  torch::jit::Module&& module) {
  std::scoped_lock<std::mutex> lk(module_mut_);
  key_ = std::move(key);
  module_ = std::move(module);
  module_.to(device_);
}

hexzpb::ModelKey FiberTorchModel::Key() const {
  std::scoped_lock<std::mutex> lk(module_mut_);
  return key_;
}

Model::Prediction FiberTorchModel::Predict(const Board& board,
                                           const Node& node) {
  Perfm::Scope ps(Perfm::Predict);
  boost::fibers::promise<Model::Prediction> promise;
  auto future = promise.get_future();
  PushRequest(PredictionRequest{
      .board = board.Tensor(node.NextTurn()),
      .action_mask = node.ActionMask().flatten(),
      .result_promise = std::move(promise),
  });
  return future.get();
}

int FiberTorchModel::PopRequests(std::vector<PredictionRequest>& buf, int n) {
  std::unique_lock<std::mutex> lock(request_mut_);
  // Wait until there's data or a fiber has left.
  request_queue_cv_.wait(
      lock, [this] { return !request_queue_.empty() || fiber_left_; });
  if (fiber_left_) {
    fiber_left_ = false;
    return 0;
  }
  int k;
  for (k = 0; k < n && !request_queue_.empty(); k++) {
    buf.push_back(std::move(request_queue_.front()));
    request_queue_.pop();
  }
  return k;
}

void FiberTorchModel::PushRequest(PredictionRequest&& request) {
  std::scoped_lock<std::mutex> lk(request_mut_);
  request_queue_.push(std::move(request));
  request_queue_cv_.notify_one();
}

void FiberTorchModel::RunGPUPipeline() {
  // Ensure this thread's perfm stats are accumulated into total stats.
  Perfm::ThreadScope perfm_thread_scope;
  ABSL_LOG(INFO) << "FiberTorchModel::GPUPipeline started";
  std::vector<PredictionRequest> batch;
  batch.reserve(batch_size_);
  while (true) {
    // Collect requests for batching
    while (batch.size() < batch_size_) {
      int k = PopRequests(batch, batch_size_ - batch.size());
      if (k == 0) {
        std::scoped_lock<std::mutex> lk(request_mut_);
        // Nothing was read => a fiber has left the building.
        // Update expected batch size.

        // ABSL_LOG(INFO) << "GPU pipelne: a fiber left. " << active_fibers_
        //                << " remain";

        if (batch_size_ > active_fibers_) {
          // Avoid waiting for a larger batch size than fibers can produce.
          batch_size_ = active_fibers_;
          //   ABSL_LOG(INFO) << "Reduced batch size to " << batch_size_;
        }
      }
    }
    if (batch.empty()) {
      ABSL_LOG(INFO) << "Nothing left to read from the queue. Exiting.";
      return;
    }
    const size_t n_inputs = batch.size();
    std::vector<torch::Tensor> boards;
    boards.reserve(n_inputs);
    std::vector<torch::Tensor> action_masks;
    action_masks.reserve(n_inputs);
    for (auto& req : batch) {
      boards.emplace_back(std::move(req.board));
      action_masks.emplace_back(std::move(req.action_mask));
    }

    std::vector<torch::jit::IValue> model_inputs = {
        torch::stack(boards).to(device_),
        torch::stack(action_masks).to(device_),
    };
    {
      Perfm::Scope perfm(Perfm::PredictBatch);
      // Need to guard access to module_ b/c it might get updated concurrently.
      std::unique_lock<std::mutex> lk(module_mut_);
      auto pred = module_.forward(model_inputs);
      lk.unlock();
      ABSL_DCHECK(pred.isTuple());
      const auto output_tuple = pred.toTuple();
      ABSL_DCHECK(output_tuple->size() == 2);
      // Read logits back to CPU / main memory.
      const auto probs =
          output_tuple->elements()[0].toTensor().exp().to(torch::kCPU);
      const auto dim = probs.sizes();
      ABSL_DCHECK(dim.size() == 2 && dim[0] == n_inputs &&
                  dim[1] == 2 * 11 * 10);
      const auto values =
          output_tuple->elements()[1].toTensor().to(torch::kCPU);
      for (int i = 0; i < n_inputs; i++) {
        batch[i].result_promise.set_value(Model::Prediction{
            .move_probs = probs.index({i}).reshape({2, 11, 10}),
            .value = values.index({i}).item<float>(),
        });
      }
      batch.clear();
    }
  }
}

ScopeGuard FiberTorchModel::RegisterThread() {
  std::scoped_lock<std::mutex> lk(request_mut_);
  active_fibers_++;
  return ScopeGuard([this] { Unregister(); });
}

void FiberTorchModel::Unregister() {
  std::scoped_lock<std::mutex> lk(request_mut_);
  ABSL_CHECK(active_fibers_ > 0);
  active_fibers_--;
  fiber_left_ = true;
  request_queue_cv_.notify_one();
}

std::string PlayoutRunner::Stats::DebugString() const noexcept {
  return absl::StrFormat("PlayoutRunner::Stats{.runs=%d, .avg=%.3f}", runs,
                         Avg());
}

PlayoutRunner::Stats RandomPlayoutRunner::Run(const Board& board, int turn,
                                              int runs) {
  PlayoutRunner::Stats stats;
  for (int i = 0; i < runs; i++) {
    Perfm::Scope perfm(Perfm::RandomPlayout);
    float r = FastRandomPlayout(turn, board);
    stats.Add(r);
  }
  aggregated_stats_.Merge(stats);
  return stats;
}

NeuralMCTS::NeuralMCTS(Model& model,
                       std::unique_ptr<PlayoutRunner> playout_runner,
                       const Config& config)
    : model_{model},
      playout_runner_{std::move(playout_runner)},
      config_{config} {}

bool NeuralMCTS::SelfplayRun(Node& root, const Board& b, bool add_noise,
                             bool run_playouts) {
  Board board(b);
  Node* n = &root;
  if (add_noise && !n->IsLeaf()) {
    // root has already been expanded before, so we can add the noise
    // directly. If this is the first time we expand root, noise is added
    // below once the predictions are available.
    n->AddDirichletNoise(kNoiseWeight, config_.dirichlet_concentration);
    add_noise = false;
  }
  // Find leaf node
  while (n->visit_count() > 0 && !n->IsLeaf()) {
    Node* child = n->MaxPuctChild();
    board.MakeMove(child->MoveTurn(), child->move());
    n = child;
  }
  if (!n->IsLeaf()) {
    // n->visit_count() == 0, but we've been here before.
    // Backprop known model prediction.
    n->Backpropagate(n->ValueP0());
    return true;  // Indicate that we re-used existing predictions.
  }
  if (n->terminal()) {
    // Nothing left to be done here, the game is over.
    n->Backpropagate(board.Result());
    return false;
  }
  // Expand leaf node. Usually it's the turn as indicated by the node.
  int next_turn = n->NextTurn();
  auto moves = board.NextMoves(next_turn);
  if (moves.empty()) {
    // Player has no valid moves left. Try if opponent can proceed.
    next_turn = 1 - next_turn;
    n->SetNextTurn(next_turn);
    moves = board.NextMoves(next_turn);
    if (moves.empty()) {
      // No player can make a move => game over.
      n->SetTerminal(true);
      n->Backpropagate(board.Result());
      return false;
    }
  }
  n->CreateChildren(moves);
  n->ShuffleChildren();  // Avoid selection bias.
  auto pred = model_.Predict(board, *n);
  predictions_count_++;
  n->SetMoveProbs(pred.move_probs);
  if (add_noise && n == &root) {
    // root has been expanded for the first time.
    n->AddDirichletNoise(kNoiseWeight, config_.dirichlet_concentration);
  }
  float value = pred.value;
  if (run_playouts) {
    auto stats =
        playout_runner_->Run(board, n->NextTurn(), config_.random_playouts);
    random_playouts_count_++;
    // 50% weight for model predictions and 50% for random playouts.
    float playout_value = (n->NextTurn() == 0 ? 1 : -1) * stats.Avg();
    value = (value + playout_value) / 2;
  }
  n->SetValue(value);
  // Backpropagate the model prediction. Need to reorient it s.t. 1 means
  // player 0 won.
  n->Backpropagate(n->ValueP0());
  return false;
}

bool NeuralMCTS::Run(Node& root, const Board& b) {
  // TODO: Check that this is all still correct. We are making a lot of
  // changes to SelfplayRun which might not all be reflected here.
  Board board(b);
  Perfm::Scope ps(Perfm::NeuralMCTS_Run);
  Node* n = &root;

  // Move to leaf node.
  auto t_start = UnixMicros();
  {
    Perfm::Scope ps(Perfm::FindLeaf);
    while (!n->IsLeaf()) {
      Node* child = n->MaxPuctChild();
      board.MakeMove(child->MoveTurn(), child->move());
      n = child;
    }
  }
  // Expand leaf node. Usually it's the turn as indicated by the node.
  int next_turn = n->NextTurn();
  auto moves = board.NextMoves(next_turn);
  if (moves.empty()) {
    // Player has no valid moves left. Try if opponent can proceed.
    next_turn = 1 - next_turn;
    n->SetNextTurn(next_turn);
    moves = board.NextMoves(next_turn);
    if (moves.empty()) {
      // No player can make a move => game over.
      n->SetTerminal(true);
      n->Backpropagate(board.Result());
      return n != &root;  // Return if we made any progress at all in this run.
    }
  }
  n->CreateChildren(moves);
  n->ShuffleChildren();  // Avoid selection bias.
  auto pred = model_.Predict(board, *n);
  n->SetMoveProbs(pred.move_probs);
  n->SetValue(pred.value);
  n->Backpropagate(n->ValueP0());
  return true;
}

std::pair<int, bool> NeuralMCTS::NumRuns(int move) const noexcept {
  if (move >= kFirstFastMove && config_.fast_move_prob > 0) {
    if (internal::UnitRandom() < config_.fast_move_prob) {
      return std::make_pair(config_.runs_per_fast_move, true);
    }
  }
  return std::make_pair(config_.runs_per_move, false);
}

absl::StatusOr<std::vector<hexzpb::TrainingExample>> NeuralMCTS::PlayGame(
    Board& board, int max_runtime_seconds) {
  Perfm::Scope ps(Perfm::PlayGame);
  std::vector<hexzpb::TrainingExample> examples;
  int64_t started_micros = UnixMicros();
  int n = 0;
  // Every game starts with player 0.
  auto root = std::make_unique<Node>(/*turn=*/0);
  float result = 0.0;
  bool game_over = false;
  const int64_t max_micros =
      max_runtime_seconds > 0
          ? started_micros +
                static_cast<int64_t>(max_runtime_seconds) * 1'000'000
          : std::numeric_limits<int64_t>::max();
  for (; n < config_.max_moves_per_game; n++) {
    int64_t move_started = UnixMicros();
    if (move_started > max_micros) {
      return absl::DeadlineExceededError(
          "max_runtime_seconds exceeded before the game was finished");
    }
    const auto [runs, is_fast_run] = NumRuns(n);
    int n_runs = 0;
    int n_reused = 0;
    bool run_playouts = !is_fast_run && config_.random_playouts > 0;
    bool add_noise = !is_fast_run && config_.dirichlet_concentration > 0;
    playout_runner_->ResetStats();
    while ((n_runs - n_reused) < runs) {
      float playout_value = 0;
      if (SelfplayRun(*root, board, add_noise, run_playouts)) {
        // Re-used previous prediction, so don't count as a full run.
        n_reused++;
      }
      n_runs++;
      add_noise = false;  // Only add noise on first run.
    }
    if (run_playouts && std::abs(playout_runner_->AggregatedStats().Avg()) >
                            config_.resign_threshold) {
      const auto [s0, s1] = board.Score();
      ABSL_LOG(INFO) << "Resigning on move " << n << " at score " << s0 << "-"
                     << s1 << " and playout result "
                     << playout_runner_->AggregatedStats().DebugString()
                     << " and root->value " << root->value()
                     << " and root->turn " << root->NextTurn();
      // Ensure the result is a whole number.
      result = std::round(playout_runner_->AggregatedStats().Avg());
      game_over = true;
      break;
    }
    // if (n < 10)
    //   WriteDotGraph(*root, "/tmp/searchtree_" + std::to_string(n) +
    //   ".dot");
    if (root->terminal()) {
      result = board.Result();
      ABSL_LOG(INFO) << "Game over after "
                     << absl::StrFormat(
                            "%.1fs",
                            static_cast<float>(UnixMicros() - started_micros) /
                                1e6)
                     << " and " << n << " moves. Final score: " << board.Score()
                     << ". Result: " << result
                     << ". Examples: " << examples.size()
                     << ". Predictions: " << PredictionsCount()
                     << ". RandomPlayouts: " << RandomPlayoutsCount();
      game_over = true;
      break;
    }

    if (!is_fast_run) {
      // Add example.
      hexzpb::TrainingExample example;
      int64_t move_ready = UnixMicros();
      example.set_unix_micros(move_ready);
      example.set_turn(root->NextTurn());
      auto enc_priors = torch::pickle_save(root->Priors());
      example.mutable_model_predictions()->mutable_priors()->assign(
          enc_priors.begin(), enc_priors.end());
      example.mutable_model_predictions()->set_value(root->value());
      example.mutable_stats()->set_duration_micros(move_ready - move_started);
      example.set_encoding(hexzpb::TrainingExample::PYTORCH);
      // The board in the example must be viewed from the perspective of the
      // player whose turn it is.
      auto enc_b = torch::pickle_save(board.Tensor(root->NextTurn()));
      example.mutable_board()->assign(enc_b.begin(), enc_b.end());
      auto enc_mask = torch::pickle_save(root->ActionMask());
      example.mutable_action_mask()->assign(enc_mask.begin(), enc_mask.end());
      auto enc_pr = torch::pickle_save(root->NormVisitCounts());
      example.mutable_move_probs()->assign(enc_pr.begin(), enc_pr.end());
      *example.mutable_model_key() = model_.Key();
      root->PopulateStats(*example.mutable_stats());
      examples.push_back(std::move(example));
    }

    const std::string stats = root->Stats();
    int turn = root->NextTurn();
    int child_idx = root->SampleChild();
    const auto& child = *root->children()[child_idx];
    const auto& move = child.move();
    // ABSL_LOG(INFO) << "#" << n << " " << move.DebugString()
    //                << (is_fast_run ? "[fast]" : "") << " (turn: " << turn
    //                << ") " << board.ShortDebugString() << " after "
    //                << (float)(UnixMicros() - started_micros) / 1000000
    //                << "s. Stats: " << stats;

    if (!is_fast_run) {
      // Update TrainingExample with information from selected child.
      auto& ex = examples.back();
      auto* mv = ex.mutable_move();
      mv->set_move(n);
      mv->set_cell_type(move.typ == Move::Typ::kFlag ? hexzpb::Field::FLAG
                                                     : hexzpb::Field::NORMAL);
      mv->set_row(move.r);
      mv->set_col(move.c);
      mv->set_player_num(turn + 1);  // 1-based

      ex.mutable_stats()->set_selected_child_q(child.Q());
      ex.mutable_stats()->set_selected_child_vc(child.visit_count());
    }

    board.MakeMove(turn, move);

    // Replace root with selected child.
    root = root->SelectChildAsRoot(child_idx);
    if (is_fast_run && config_.random_playouts > 0) {
      // Gnarf, this "fast run" code proliferates...
      // If we make a fast run and use random playouts,
      // we should not re-use the .value of child nodes
      // as it interferes badly with value updates done due to random
      // playouts: the .value of a fast run was generated using only model
      // predictions. For simplicity, just don't re-use the tree at all.
      root = std::make_unique<Node>(root->NextTurn());
    } else {
      root->ResetTree();
    }
  }
  if (game_over) {
    for (auto& ex : examples) {
      ex.set_result(ex.turn() == 0 ? result : -result);
    }
  }
  return examples;
}

absl::StatusOr<std::unique_ptr<Node>> NeuralMCTS::SuggestMove(
    int player, const Board& board, int think_time_millis) {
  int64_t started_micros = UnixMicros();
  auto root = std::make_unique<Node>(player);
  const int64_t max_micros =
      started_micros + static_cast<int64_t>(think_time_millis) * 1000;
  for (int n = 0; n < config_.runs_per_move; n++) {
    if (UnixMicros() > max_micros) {
      break;
    }
    if (!Run(*root, board)) {
      break;
    }
  }
  if (root->IsLeaf() || root->NextTurn() != player) {
    // Run may flip the turn of the root node if it finds that only
    // the opponent can make a valid move.
    return absl::InvalidArgumentError("Player has no valid moves left.");
  }
  return root;
}

}  // namespace hexz
