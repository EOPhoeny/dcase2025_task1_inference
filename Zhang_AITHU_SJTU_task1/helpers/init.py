import random

import numpy as np
import torch


def worker_init_fn(wid):
    """
    This function is passed to Pytorch dataloader and makes sure
    that python random, numpy and torch are seeded appropriately.
    """
    seed_sequence = np.random.SeedSequence(
        [torch.initial_seed(), wid]
    )

    # 尝试使用 uint64 类型的种子
    # PyTorch 期望的种子通常是 long (64位)
    to_seed = spawn_get(seed_sequence, 2, dtype=int)
    torch.random.manual_seed(to_seed)

    np_seed = spawn_get(seed_sequence, 2, dtype=np.ndarray)
    np.random.seed(np_seed)

    py_seed = spawn_get(seed_sequence, 2, dtype=int)
    random.seed(py_seed)


def spawn_get(seedseq, n_entropy, dtype):
    child = seedseq.spawn(1)[0]
    # generate_state 已经返回 uint32 类型
    state = child.generate_state(n_entropy, dtype=np.uint32)

    if dtype == np.ndarray:
        return state
    elif dtype == int:
        # 使用 numpy 的 uint64 来组合这些 uint32 值
        # 确保结果在 uint64 的范围内，并且进行位移操作
        state_as_uint64 = np.uint64(0)
        for shift, s in enumerate(state):
            # 将每个 uint32 元素 s 强制转换为 uint64 后进行位移
            # 这样可以在 numpy 层面处理大整数，避免 Python 整数在传递时潜在的溢出
            state_as_uint64 += np.uint64(s) << (32 * shift)
            
        # 返回 Python 整数，因为 torch.random.manual_seed 和 random.seed 期望 Python 整数
        # 但我们确保它在 uint64 范围内
        return int(state_as_uint64) 
    else:
        raise ValueError(f'not a valid dtype "{dtype}"')
