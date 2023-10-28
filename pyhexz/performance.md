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

From https://pythonspeed.com/articles/docker-performance-overhead/ I learnt that sometime
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
