# ML Engine Improvement Plan

## Goal

Build a Flagz engine that is substantially stronger than the existing
`res10` model, with progress demonstrated by repeatable paired evaluations
rather than training loss alone.

The long-term target remains:

- decisively beat the existing `res10` baseline;
- beat the Go MCTS player even when it receives a much larger search budget;
- support a sustainable self-play → train → evaluate loop.

## Where We Stand

### No new model has been trained

No training update has been performed during this work. In particular, we have
not trained a new checkpoint on the old replay data.

An isolated candidate named `res10-r4-cp63` was previously initialized:

- its checkpoint 0 is an exact copy of `res10:63`;
- it has an isolated replay file seeded with 1,025,000 examples;
- it has no checkpoint 1 and has never received worker examples.

This candidate will not be used for the main experiment because checkpoint 62
now has better supporting evidence. It will also not be deleted without an
explicit decision.

No `res10-r4-cp62` candidate currently exists.

### Repository changes already committed

The branch contains these relevant improvements:

- fixed, seeded starting positions for repeatable model evaluation;
- paired evaluation with each model playing both sides;
- concurrent evaluation games;
- paired-score confidence intervals;
- bounded training: 25 optimizer batches per 25,000 new examples;
- a 1,048,576-example replay window;
- persistent Adam optimizer state and global training step;
- isolated candidate initialization;
- local virtualenv support for the training server;
- reproducible replay sampling using 256-example contiguous reads, combining
  16 randomly ordered regions per optimizer batch;
- sampled replay ranges and estimated fresh-example counts persisted with each
  checkpoint;
- finite-loss, finite-gradient, and finite-parameter publication guards;
- corrected per-layer gradient diagnostics and detailed training timing.

The complete Python and Go test suites passed after the training-soundness
changes:

- Python: 17 passed, 2 intentional performance-test skips;
- Go: all packages passed.

## Findings

### Checkpoint 63 is not degenerated, but checkpoint 62 is better

The concern that the existing model might have collapsed or developed
vanishing gradients was tested directly.

Checkpoint 63:

- has finite, non-zero gradients throughout the network;
- has healthy gradient signals in early layers and both output heads;
- improves held-out loss after one optimizer step;
- therefore shows no evidence of vanishing gradients or an untrainable model.

However, checkpoint 63 regressed relative to checkpoint 62 on a fixed replay
sample. The clearest regression is in the value head:

| Metric | Checkpoint 62 | Checkpoint 63 |
| --- | ---: | ---: |
| Value MSE | ~0.30 | ~0.38 |
| Value-sign accuracy | ~89.5% | ~86.5% |
| Value bias | ~+0.03 | ~+0.17 |
| Policy cross-entropy | ~2.02 | ~2.03 |

A fixed 4,096-example sweep across all checkpoints 0–63 found checkpoint 62
had the best combined policy/value loss. Checkpoint 59 was marginally best on
policy loss alone, but materially worse on value prediction.

The likely failure mode is late-checkpoint oscillation or overfitting, not
model degeneration.

### Small playing-strength screens support checkpoint 62

These were deliberately small smoke evaluations, not promotion tests:

| Match | Result |
| --- | --- |
| `res10:62` vs `res10:63` | 11–4, with 1 draw |
| `res10:62` vs `res10:57` | 11–5 |

Together with the offline metrics, this is enough evidence to use checkpoint
62 as the warm start. It is not enough evidence to claim checkpoint 62 is a
new champion.

### Warm-starting is the sensible first experiment

Starting from checkpoint 62 is preferable to random initialization for the
first controlled experiment because:

- it already contains useful game knowledge;
- it is demonstrably trainable;
- it is currently stronger than the last checkpoint;
- it lets us test the corrected training regime quickly;
- random initialization would require much more self-play and compute before
  producing a meaningful comparison.

Training from scratch remains a useful later control if continued training
fails, but there is currently no evidence that it is necessary.

### Measured self-play performance

All measurements used 800 MCTS runs per move and completed games only.
Dry-run mode prevented examples from being stored or triggering training.

| Worker configuration | Games | Examples | Runtime | Completed examples/s |
| --- | ---: | ---: | ---: | ---: |
| `cuda@1:16:16` | 16 | 1,156 | 55.8 s | 20.7 |
| `cuda@1:64:64` | 64 | 4,878 | 184.5 s | 26.4 |
| `cuda@4:128:256` | 512 | 38,445 | 522.8 s | 73.5 |

The production-shaped configuration is clearly the throughput winner. It
sustains roughly 55,000–59,000 predictions/s while fully loaded.

Its tradeoff is latency:

- the first of 512 games completed after about 270 seconds;
- the complete wave took about 523 seconds;
- a 120-second worker run can therefore do substantial GPU work but deliver
  zero completed games;
- unfinished games are correctly discarded when the configured runtime ends.

The previously observed ~102 examples/s is a short-window, in-flight rate.
The measured end-to-end rate for a completed 512-game wave is 73.5 examples/s.

At that rate, generating 25,000 examples requires about 5.7 minutes of
aggregate work. From a synchronized cold start, the trigger will occur later
because the completed games arrive in a wave.

## Training Regime to Test

The intended loop remains simple:

1. Workers generate complete self-play games with the current checkpoint.
2. Complete games are uploaded; unfinished games are not.
3. After 25,000 new examples, the server performs one bounded update.
4. The update creates the next checkpoint.
5. Workers continue with the new checkpoint.
6. Periodically, `nbench2` evaluates playing strength.
7. Repeat.

The first experiment will use:

| Setting | Value |
| --- | --- |
| Warm start | `res10:62` |
| Candidate | `res10-r4-cp62` |
| Replay seed | newest 1,025,000 existing examples |
| Training trigger | 25,000 new completed examples |
| Replay window | newest 1,048,576 examples |
| Batches per trigger | 25 |
| Batch size | 4,096 |
| Replay sampling block | 256 examples |
| Replay sampling seed | 1 + source checkpoint |
| Examples trained per trigger | 102,400 |
| Effective replay reuse | 4.096 |
| Optimizer | Adam with persistent state |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Epoch cap | 1 |
| Self-play worker | `cuda@4:128:256` |
| Worker runtime | explicitly configured per run |
| Training/self-play GPU contention | worker suspends while training |

The replay seed makes this a continuation experiment, not training from
scratch. The first checkpoint will be trained on a shuffled sample of the
replay window after at least 25,000 newly generated examples have arrived.

## Next Experiment

### Step 1: Create the isolated checkpoint-62 candidate

Create `res10-r4-cp62` with:

- checkpoint 0 copied exactly from `res10:62`;
- its own replay HDF5 file;
- 1,025,000 newest examples copied from the existing replay;
- no changes to `res10`, `res10-r4-cp63`, or their replay data.

Verify before proceeding:

- candidate checkpoint 0 weights exactly match `res10:62`;
- replay contains exactly 1,025,000 aligned examples;
- no checkpoint 1 exists;
- no optimizer state exists yet.

### Step 2: Run exactly one self-play/training cycle

Start the candidate training server and one production-shaped CUDA worker.
The worker runtime will be set long enough to complete a wave and will suspend
while the shared GPU is training.

Stop after checkpoint 1 has been created. Do not allow an accidental second
training trigger.

Record:

- time to receive 25,000 complete examples;
- complete games and examples accepted;
- setup, training, and checkpoint-serialization time;
- all 25 policy, value, and combined losses;
- per-layer and global gradient/parameter ratios;
- replay ranges and estimated fresh-example count sampled;
- data-loading, device-transfer, compute, and checkpoint times;
- optimizer global step and persisted state;
- checkpoint file integrity;
- peak GPU memory if readily available.

Failure gates:

- non-finite loss or gradients;
- zero gradients in a substantial part of the model;
- anomalously large gradient/parameter ratios;
- checkpoint cannot be reloaded;
- optimizer state is not persisted;
- more than one checkpoint is created;
- examples from unfinished games are stored.

If any failure gate is hit, stop and diagnose before generating more data.

### Step 3: Evaluate checkpoint 1 before continuing training

First run a cheap paired screen using the fixed starting-position set:

- candidate checkpoint 1 versus candidate checkpoint 0;
- both models play both sides from every selected starting position;
- games run concurrently;
- report wins, draws, score rate, and paired confidence interval.

Also run the fixed offline replay evaluation to check:

- policy cross-entropy;
- value MSE;
- value-sign accuracy;
- value bias;
- finite/non-zero gradients.

Decision:

- if checkpoint 1 is clearly worse or numerically unhealthy, stop;
- if results are neutral but healthy, run a second 25k cycle before judging;
- if results are positive, continue for a small sequence of checkpoints and
  evaluate the trend;
- do not promote a model based on training loss alone.

### Step 4: Establish a short learning curve

If the first checkpoint is healthy, repeat the exact regime for a small,
predefined number of triggers. Evaluate at regular checkpoints against:

- the checkpoint-62 warm start;
- `res10:63`, because it is the historical external baseline;
- the strongest preceding candidate checkpoint.

This answers the main question: does the corrected data-generation and replay
regime produce monotonically stronger play, or merely move offline losses?

Only after a positive short learning curve should the experiment run
unattended for a long period.

## Evaluation Policy

Evaluation uses fixed starting positions to reduce noise and paired games to
remove first/second-player bias. Games are concurrent so evaluation does not
become a sequential bottleneck.

There are two evaluation levels:

1. **Screening:** small, fast paired match used to reject obvious regressions.
2. **Promotion:** larger paired match with a confidence interval tight enough
   to support a model-selection decision.

The exact promotion game count should be selected from the observed variance
and desired confidence width, rather than an arbitrary number. A checkpoint
is not called stronger merely because it wins a tiny match.

## What We Will Not Change Yet

The first experiment is intended to validate the learning loop. It will not
also introduce:

- board reflections or other augmentation;
- a new network architecture;
- squeeze-and-excitation blocks;
- a new value or score-margin head;
- AdamW or SGD;
- a learning-rate scheduler;
- value-loss reweighting;
- MCTS temperature or FPU changes;
- training from random initialization.

Those may be valuable, but changing them now would make it impossible to know
whether the corrected replay/training regime works.

## Follow-up Experiments, in Priority Order

Once the base loop shows measurable improvement:

1. Tune learning rate and optimizer using short, matched training runs.
2. Tune replay mixture/window and examples trained per 25k fresh examples.
3. Tune value-loss weight, because the current model's largest observed
   regression was in value prediction.
4. Tune self-play exploration and move-temperature schedule.
5. Add symmetry augmentation if Flagz's exact board/rule symmetries are
   verified and the transformed policy/action masks are tested.
6. Evaluate architecture changes, beginning with changes that improve global
   board context.
7. Compare the best warm-start run with a properly budgeted from-scratch run.

Every experiment should change one major variable, use the same evaluation
positions and budgets, and preserve its model/replay data under a distinct
candidate name.

## Immediate Decision Needed

The code and measurement gates are ready. The next action that changes data is
creating `$HOME/data/hexz-models/models/flagz/res10-r4-cp62` and then running
one bounded self-play/training cycle. Nothing else needs to be optimized before
that experiment.
