
for t in 1 2 4 8 16; do
	for f in 64 128 256; do
		for b in 128 256 512; do
			echo "Config: threads=$t fibers=$f batch_size=$b" | tee -a worker.log
			./worker --worker_threads=$t --training_server_addr=unused --max_runtime_seconds=10 --device=cuda --fibers_per_thread=$f --prediction_batch_size=$b --runs_per_move=800 --dry_run --local_model_path=$HOME/tmp/hexz-models/models/flagz/resus/checkpoints/0/scriptmodule.pt 2>&1 | tee -a worker.log
		done
	done
done


