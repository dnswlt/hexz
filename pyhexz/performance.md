# Performance measurement log

## 2023-10-27

What's going on in Cloud Run? `run_find_leaf` is horrensibly slow!?

```
2023-10-26 22:14:35.003 CEST
method                   total_sec      count      ops/s
2023-10-26 22:14:35.003 CEST
CBoard.__init__             0.607s      64801   106725.1
2023-10-26 22:14:35.003 CEST
NeuralMCTS.predict         55.127s       8371      151.8
2023-10-26 22:14:35.003 CEST
NeuralMCTS.__init__         0.252s          1        4.0
2023-10-26 22:14:35.003 CEST
run_find_leaf              79.238s      64800      817.8
2023-10-26 22:14:35.003 CEST
PurePyBoard.next_moves      1.943s     120658    62084.8
2023-10-26 22:14:35.003 CEST
NeuralMCTS.run            149.082s      64800      434.7
2023-10-26 22:14:35.003 CEST
CBoard.make_move            0.001s         80    57971.0
```

From <https://pythonspeed.com/articles/docker-performance-overhead/> I learnt that sometime
Docker security features are to blame.

I can reproduce a significant speedup for `run_find_leaf` on my nuc when running Docker with
`--security-opt seccomp=unconfined`: 1124.3 ops/s vs. 2197.1 ops/s, almost a 2x speedup:

```
(pyhexz) dw@nuc:~/git/github.com/dnswlt/hexz/pyhexz$ docker run  -e PYTHONUNBUFFERED=1 -e HEXZ_TRAINING_SERVER_URL=http://nuc:8080   europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/worker:latest
cuda available: False
mps available: False
torch version: 2.0.1
2023-10-27 07:04:23,473 INFO SelfPlayWorker Running with config=WorkerConfig(training_server_url='http://nuc:8080', device='cpu', max_seconds=60, runs_per_move=800, http_client_timeout=1.0)
2023-10-27 07:04:23,579 INFO SelfPlayWorker Server at http://nuc:8080 is using model abel:0.
Iteration 0 @2.387s: visit_count:799  move:(0, 2, 9, 1) player:0 score:(0.0, 0.0)
Iteration 1 @4.567s: visit_count:1598  move:(0, 0, 6, 1) player:1 score:(0.0, 0.0)
Iteration 2 @6.640s: visit_count:2397  move:(0, 7, 4, 1) player:0 score:(0.0, 0.0)
Iteration 3 @8.606s: visit_count:3196  move:(0, 6, 1, 1) player:1 score:(0.0, 0.0)
Iteration 4 @10.469s: visit_count:3995  move:(0, 10, 8, 1) player:0 score:(0.0, 0.0)
Iteration 10 @20.232s: visit_count:8789  move:(1, 10, 7, 1.0) player:0 score:(4.0, 2.0)
Iteration 20 @39.501s: visit_count:11605  move:(1, 9, 7, 1.0) player:0 score:(11.0, 10.0)
Iteration 30 @58.790s: visit_count:12634  move:(1, 10, 2, 4.0) player:0 score:(22.0, 17.0)
Iteration 40 @72.414s: visit_count:12831  move:(1, 5, 6, 3.0) player:0 score:(35.0, 33.0)
Iteration 50 @80.916s: visit_count:12768  move:(1, 8, 5, 1.0) player:0 score:(50.0, 49.0)
Iteration 60 @84.549s: visit_count:13449  move:(1, 10, 6, 2.0) player:0 score:(59.0, 60.0)
Done in 84.963s after 65 moves. Final score: (64.0, 64.0).
method                   total_sec      count      ops/s
CBoard.__init__             0.308s      52801   171195.6
NeuralMCTS.predict         29.435s      11710      397.8
NeuralMCTS.__init__         0.004s          1      222.6
run_find_leaf              46.964s      52800     1124.3
PurePyBoard.next_moves      0.982s      93149    94873.1
NeuralMCTS.run             84.802s      52800      622.6
CBoard.make_move            0.001s         65    51669.3
(pyhexz) dw@nuc:~/git/github.com/dnswlt/hexz/pyhexz$ docker run --security-opt seccomp=unconfined -e PYTHONUNBUFFERED=1 -e HEXZ_TRAINING_SERVER_URL=http://nuc:8080   europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/worker:latest
cuda available: False
mps available: False
torch version: 2.0.1
2023-10-27 07:06:10,395 INFO SelfPlayWorker Running with config=WorkerConfig(training_server_url='http://nuc:8080', device='cpu', max_seconds=60, runs_per_move=800, http_client_timeout=1.0)
2023-10-27 07:06:10,464 INFO SelfPlayWorker Server at http://nuc:8080 is using model abel:0.
Iteration 0 @1.813s: visit_count:799  move:(0, 2, 9, 1) player:0 score:(0.0, 0.0)
Iteration 1 @3.476s: visit_count:1081  move:(0, 0, 6, 1) player:1 score:(0.0, 0.0)
Iteration 2 @5.239s: visit_count:1880  move:(0, 7, 4, 1) player:0 score:(0.0, 0.0)
Iteration 3 @6.848s: visit_count:1080  move:(0, 6, 1, 1) player:1 score:(0.0, 0.0)
Iteration 4 @8.468s: visit_count:1879  move:(0, 6, 9, 1) player:0 score:(0.0, 0.0)
Iteration 10 @18.108s: visit_count:1752  move:(1, 7, 5, 1.0) player:0 score:(3.0, 2.0)
Iteration 20 @33.238s: visit_count:1867  move:(1, 8, 3, 2.0) player:0 score:(15.0, 11.0)
Iteration 30 @46.896s: visit_count:1591  move:(1, 6, 5, 1.0) player:0 score:(25.0, 20.0)
Iteration 40 @59.056s: visit_count:1824  move:(1, 6, 7, 2.0) player:0 score:(34.0, 34.0)
Iteration 50 @68.426s: visit_count:1576  move:(1, 5, 6, 3.0) player:0 score:(51.0, 42.0)
Iteration 60 @74.932s: visit_count:2387  move:(1, 8, 8, 2.0) player:0 score:(64.0, 55.0)
Iteration 70 @79.397s: visit_count:1859  move:(1, 3, 7, 2.0) player:0 score:(78.0, 69.0)
Iteration 80 @80.994s: visit_count:1950  move:(1, 9, 3, 2.0) player:0 score:(92.0, 78.0)
Done in 81.055s after 82 moves. Final score: (92.0, 79.0).
method                   total_sec      count      ops/s
CBoard.__init__             0.382s      66401   173816.4
NeuralMCTS.predict         43.646s      19272      441.5
NeuralMCTS.__init__         0.004s          1      251.4
run_find_leaf              30.222s      66400     2197.1
PurePyBoard.next_moves      1.032s     113739   110228.9
NeuralMCTS.run             80.875s      66400      821.0
CBoard.make_move            0.001s         82    64012.5
```

2023-10-28 mac M1 Pro with new hexc.py implementation:

Done in 48.210s after 53 moves. Final score: (68.0, 73.0).
method                   total_sec      count      ops/s
NeuralMCTS.__init__         0.000s          2     7168.5
CBoard.__init__             0.133s      83202   624330.3
run_find_leaf              12.118s      83200     6865.7
PurePyBoard.next_moves      0.842s     103783   123188.5
NeuralMCTS.predict         74.413s      71313      958.3
NeuralMCTS.run             90.630s      83200      918.0
CBoard.make_move            0.001s        102   185118.0

## 2023-10-31

<https://github.com/gperftools/gperftools/issues/532>

Neither gprof nor gperftools yield any reasonable profiling results:

```
(pprof) top20
Showing nodes accounting for 34.01s, 93.15% of 36.51s total
Dropped 351 nodes (cum <= 0.18s)
Showing top 20 nodes out of 71
      flat  flat%   sum%        cum   cum%
    17.51s 47.96% 47.96%     17.51s 47.96%  omp_get_num_procs
     6.27s 17.17% 65.13%      6.27s 17.17%  mkl_blas_avx2_sgemm_kernel_0
     3.45s  9.45% 74.58%      3.45s  9.45%  mkl_blas_avx2_sgemm_kernel_0_b0
     2.39s  6.55% 81.13%      3.60s  9.86%  at::native::(anonymous namespace)::unfolded2d_copy(float*, float*, long, long, long, long, long, long, long, long, long, long, long)::{lambda(long, long)#1}::operator()
     1.23s  3.37% 84.50%      1.23s  3.37%  __nss_database_lookup
     0.97s  2.66% 87.15%      0.97s  2.66%  mkl_blas_avx2_sgemm_scopy_right4_ea
     0.57s  1.56% 88.72%      0.57s  1.56%  mkl_blas_avx2_sgemm_scopy_down24_ea
     0.50s  1.37% 90.08%      0.57s  1.56%  gblock_by_k_omp
```

## 2023-11-02

C++ worker in Docker:

```
scope              total_time      count        ops/s
NeuralMCTS::Run      107.590s      70401      654.342
Predict              104.542s      49316      471.735
FindLeaf               2.510s      70401    28048.097
MaxPuctChild           2.001s    1170674   585037.079
Puct                   1.328s   16254895 12239552.417
MakeMove               0.440s    1239104  2815945.426
NextMoves              0.080s      92847  1159550.630
```

Outside Docker

```
scope              total_time      count        ops/s
NeuralMCTS::Run      111.378s      72001      646.454
Predict              109.295s      55952      511.935
FindLeaf               1.600s      72001    44999.839
MaxPuctChild           1.300s     644303   495758.242
Puct                   0.904s   10161038 11245034.907
MakeMove               0.271s     688451  2539308.996
NextMoves              0.082s      88679  1086000.210
```

Nice, they're essentially running equally fast.

## 2023-11-03

First results from Cloud Run, unoptimized and essentially unconfigured (w.r.t. CPU and RAM):

2023-11-03 20:06:46.375 CET
scope              total_time      count        ops/s
NeuralMCTS::Run      299.430s      77601      259.162
Predict              293.611s      62069      211.399
FindLeaf               4.496s      77601    17260.590
MaxPuctChild           3.651s     686922   188147.637
Puct                   2.781s   11025607  3963939.303
MakeMove               0.716s     699469   976867.984
NextMoves              0.339s      93273   275477.205

## 2023-11-04

With 2 CPU per worker we're indeed twice as fast:

scope              total_time      count        ops/s
NeuralMCTS::Run      299.116s     146273      489.018
Predict              295.952s     140748      475.578
FindLeaf               2.110s     146273    69332.292
MaxPuctChild           1.640s     416367   253867.305
Puct                   1.217s    8186561  6729533.664
MakeMove               0.409s     438087  1071685.203
NextMoves              0.241s     153626   636929.772

The ml-server also does a decent job:

[2023-11-04 23:39:02,598] INFO in training: Finished training batch of size 1024 for 7 epochs in 20.4s.

## 2023-11-10

Added two channels (now at (11, 11, 10)) for remaining flags. Also added action_mask to ignore invalid moves in policy.

```
I1110 21:54:36.649133       1 worker_main.cc:177] Worker started with Config(training_server_url: 'http://10.172.0.3', local_model_path: '', runs_per_move: 1000, runs_per_move_gradient: -0.010, max_moves_per_game: 200, max_runtime_seconds: 300, max_games: -1, uct_c: 5.000, dirichlet_concentration: 0.350)
```

2023-11-10 22:59:37.946 CET
scope              total_time      count        ops/s
NeuralMCTS::Run      299.278s     155683      520.195
Predict              295.137s     151481      513.256
NextMoves              1.833s     168750    92047.441
FindLeaf               1.501s     155683   103734.353
MakeMove               1.005s     435675   433635.199
MaxPuctChild           0.442s     422990   957970.056

[2023-11-10 22:06:16,053] INFO in training: Finished training batch of size 1024 for 7 epochs in 20.6s.

Tree reuse and Dirichlet noise:

<https://ai.stackexchange.com/questions/36808/does-the-alphazero-algorithm-keep-the-subtree-statistics-after-each-move-during>
<https://github.com/leela-zero/leela-zero/issues/538>
<https://arxiv.org/pdf/1902.10565.pdf> (KataGo)

## 2023-11-15

Using jemalloc in Docker on NUC:

scope           total_time      count        ops/s
Predict            60.557s      27700      457.423
NextMoves           0.293s      31365   107025.721
MakeMove            0.124s      91051   735719.748
MaxPuctChild        0.050s      87181  1746825.495

## 2023-11-18

Interesting discussion about setting the right initial Q value and tweaking the p_uct:
<https://ai.stackexchange.com/questions/25939/alpha-zero-does-not-converge-for-connect-6-a-game-with-huge-branching-factor>

Leela Zero's Q value settings:
<https://lczero.org/play/flags/#:~:text=%E2%80%9CFirst%20Play%20Urgency%E2%80%9D%20changes%20search,absolute%E2%80%9D%20directly%20uses%20that%20value>.

And its implementation:
<https://github.com/leela-zero/leela-zero/blob/next/src/UCTNode.cpp#L270>

## 2023-11-27

Crazy stuff: in the Debug build, random playouts are MUCH slower:

Debug:

scope            total_time      count        ops/s
RandomPlayout      198.070s    4331900    21870.554
Predict            100.392s      94538      941.687
MakeMove             0.885s     530494   599324.670
NextMoves            0.789s      95595   121204.863
MaxPuctChild         0.537s     508754   947060.179

Release:

scope            total_time      count        ops/s
Predict            237.454s     251300     1058.309
RandomPlayout       59.917s   11280000   188259.737
NextMoves            0.773s     251300   324949.740
MakeMove             0.594s    1417160  2387065.939
MaxPuctChild         0.398s    1354861  3406033.806

## 2023-12-05

In single-threaded mode, Predict runs 3-4 times faster on CPU than on MPS:

CPU:

scope            total_time      count        ops/s
PlayGame            60.628s          1        0.016
Predict             53.048s      15100      284.646
RandomPlayout        7.288s     720000    98791.874
MakeMove             0.112s      94085   840112.474
NextMoves            0.058s      15100   259301.319
MaxPuctChild         0.032s      90728  2845707.531

MPS:

scope            total_time      count        ops/s
PlayGame            60.480s          1        0.017
Predict             44.480s      41200      926.258
RandomPlayout       15.291s    1980000   129486.310
NextMoves            0.162s      41200   254693.001
MakeMove             0.133s     239363  1804754.206
MaxPuctChild         0.086s     228045  2643199.103
