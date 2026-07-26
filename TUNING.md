# Hexz ML-Player Tuning & Progress Log

## Objective & Problem Statement

### Problem Statement
The baseline ML player (`res10` checkpoint 63) demonstrates competitive playing strength, but is not unbeatable:
- It can still be outplayed by the reference Go-based MCTS engine when the Go engine is given deeper search trees (e.g. 32,000 iterations vs 3,200).
- It occasionally makes tactical or endgame errors under heavy search pressure.

### Primary Goal
**Build a next-generation ML engine that decisively crushes both the baseline `res10:63` model and the reference Go MCTS engine across all iteration regimes.**

### Success Criteria
- **>90% Win Rate** against the reference Go MCTS engine (even when the Go engine is granted 10x higher iteration budgets).
- **>80% Win Rate** in direct head-to-head evaluation matches against the baseline `res10` checkpoint 63 model.

---

## Environment Setup & Machine Spec

- **Hardware**: NVIDIA GeForce RTX 4080 (16GB VRAM), 28-core CPU.
- **OS**: Ubuntu 26.04 LTS.
- **NVIDIA Container Toolkit Setup**: Added the official NVIDIA apt repository and runtime configuration:
  ```bash
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- **Python Environment**: Dedicated virtual environment located at `pyhexz/.venv` with PyTorch 2.13 (CUDA 13 support), `grpcio-tools`, `h5py`, `flask`, `gunicorn`, `tensorboard`, `pytest`.
- **Baseline Model Target**: `res10` (10 ResNet blocks, 128 filters) at **checkpoint 63**, located in `/home/dw/data/hexz-models/models/flagz/res10/checkpoints/63`.

---

## Log of Setup & Configuration Changes

### 1. Protobuf Generation Script (`scripts/run_protoc.sh`)
- Updated `scripts/run_protoc.sh` to automatically detect virtualenv Python (`pyhexz/.venv/bin/python3`) without requiring `$CONDA_PYTHON_EXE`.
- Verified cleanly generated Python protobuf stubs: `hexz_pb2.py`, `hexz_pb2_grpc.py`, `nbench_pb2.py`, `nbench_pb2_grpc.py`.

### 2. CUDA Worker Docker Build (`cpp/Dockerfile.worker-cuda`)
- Fixed `protoc` include path flag to explicitly target `proto/hexz.proto` and `proto/health.proto` matching `scripts/run_protoc.sh cpp`.
- Enabled CUDA worker container image build (`hexz-worker-cuda:latest`).

### 3. Dedicated CUDA CPUServer Docker Build (`cpp/Dockerfile.cpuserver-cuda`)
- Created dedicated Dockerfile `cpp/Dockerfile.cpuserver-cuda` for building the `cpuserver` binary (`cpuserver_main` target).
- Used for `nbench` evaluation games to serve model predictions (`res10:63`) on GPU (`hexz-cpuserver-cuda:latest`).

---

## Standard Execution & Benchmark Commands

### 1. Running the Local Training Pipeline (Self-Play & Training)
- **Start Local Training Server**:
  ```bash
  bash scripts/training_server_local.sh
  ```
  Starts Flask/gRPC server loading model checkpoint 63 from `/home/dw/data/hexz-models`, listening on port `:8080` (HTTP) and `:50051` (gRPC).

- **Start CUDA Worker Container (Self-Play Data Generation)**:
  ```bash
  bash scripts/worker_docker_cuda.sh
  ```
  Launches `hexz-worker-cuda:latest` with `--gpus all` and `cuda@4:128:256` worker spec. Streams self-play game examples to the training server at **~59,000 predictions/s**.

### 2. Running Model Evaluation / Benchmarks (`nbench`)
- **Run Go Reference MCTS vs. GPU ML Model**:
  ```bash
  bash scripts/nbench_mcts.sh <go_iterations> <ml_iterations> <num_games>
  # Example: 5 games of Go MCTS (32k iters) vs ML Model res10:63 (3.2k iters on GPU):
  bash scripts/nbench_mcts.sh 32000 3200 5
  ```
  Starts `hexz-cpuserver-cuda:latest` in Docker listening on `:50071` with model `res10:63` on GPU, and runs `go run ./cmd/nbench` to pit Go reference MCTS (`P1`) against the ML player (`P2`). Writes output logs to `log/nbench_mcts.log`.

---

## Experiments & Benchmarks

| Exp ID | Model Architecture | Hyperparameters / Changes | Performance Metrics | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | ResNet 10 blocks, 128 filters (`res10:63`) | Original Adam fixed LR 1e-3, 7 epochs/trigger | **~59,000 predictions/s**, **~102 examples/s** on RTX 4080 (`cuda@4:128:256`) | Verified & Active |

---

## Proposed ML-Player Improvements & Optimization Roadmap (initial draft by Gemini)

### 1. Neural Network Architecture Upgrades (`pyhexz/src/pyhexz/model.py`)
- **Squeeze-and-Excitation (SE-ResNet) Blocks**: Integrate SE channel attention (`nn.AdaptiveAvgPool2d` + FC squeeze/expansion) into `ResidualBlock` to enable global board interaction across separated hex cells.
- **Global Average + Max Pooling Value Head**: Replace direct flattening of $11 \times 10$ feature maps in `value_head` with combined Global Average & Max Pooling to reduce spatial overfitting and improve endgame value predictions.
- **Hex Spatial Geometry Awareness**: Add row-parity coordinate channels (staggered odd/even row neighbor encoding) to improve $3 \times 3$ Conv feature representation on hexagonal boards.

### 2. Training Pipeline Optimization (`pyhexz/src/pyhexz/training.py` & `config.py`)
- **Optimizer Upgrade**: Switch from standard Adam ($10^{-3}$) to **AdamW** or **SGD with Momentum** ($m=0.9$).
- **Learning Rate Decay**: Add **CosineAnnealingLR** or **StepLR** scheduler ($10^{-3} \to 10^{-4} \to 10^{-5}$).
- **Replay Buffer & Epoch Count**: Reduce `num_epochs` from 7 to 1 per trigger and lower `training_trigger_threshold` (10,000–25,000 examples) to prevent over-fitting on older data and update self-play models faster.
- **Value Loss Scaling**: Increase value loss weight $c_{\text{val}} \in [1.5, 2.0]$ in `loss = pr_loss + c_val * val_loss` to accelerate MCTS node evaluation accuracy.

### 3. Self-Play & MCTS Search Enhancements (`cpp/mcts.cc`)
- **Move Temperature Decay**: Use exploration temperature $\tau = 1.0$ for the first 10–12 moves, then decay to $\tau \to 0$ (greedy selection) for the remainder of the game to eliminate random blunders in endgame training samples.
- **Dynamic First-Play Urgency (FPU)**: Replace static Q penalty with dynamic FPU ($Q_{\text{FPU}} = Q - c_{\text{fpu}} \sqrt{P(s,a)}$) to prioritize untried moves proportional to prior probability.

### 4. Auxiliary Training Objectives
- **Score Margin Lead Head**: Predict final score difference ($\text{Score}_{\text{P1}} - \text{Score}_{\text{P2}}$) alongside binary win/loss outcome to provide denser learning signals per game.


