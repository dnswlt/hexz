"""This file contains a very simple performance comparison between
regular model forward steps and AMP-enabled (automatic mixed precision)
forward steps.

To run the comparison, you need to set the action mask to a smaller value
(-1e3 instead of -1e32):

    policy = policy.where(action_mask.flatten(1), torch.full_like(policy, -1e3))

Results from RTX 4080 look promising, the APM run takes ~40% less time:

$ python -c 'import pyhexz.apm as m; m.run_all()'

run(n=100, apm_enabled=False):
Total execution time = 1.099 sec
Max memory used by tensors = 1157705728 bytes

run(n=100, apm_enabled=True):
Total execution time = 0.680 sec
Max memory used by tensors = 600931328 bytes

"""

import torch, time, gc

from pyhexz.model import HexzNeuralNetwork



# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))


def run(n=1, apm_enabled=False):
    bs = torch.rand((512, 11, 11, 10)).to('cuda')
    ms = (torch.rand((512, 2, 11, 10)) < 0.5).to('cuda')
    m = HexzNeuralNetwork(blocks=10, model_type="resnet").to('cuda')
    start_timer()
    for i in range(n):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=apm_enabled):
            m(bs, ms)
    end_timer_and_print(f"run(n={n}, apm_enabled={apm_enabled}):")

def run_all():
    run(100, False)
    run(100, True)