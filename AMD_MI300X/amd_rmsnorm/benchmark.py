from itertools import product

import torch
import tabulate
import numpy as np

import implementations as I

I.load_all()

dtypes = [torch.float, torch.float16, torch.bfloat16]
shapes = [(1, 4096, 8192)]
eps = 1e-6

warmup_runs = 3
num_runs = 100


def make_events():
    return [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(num_runs)]


def make_cache():
    return torch.empty((int(256e6),), dtype=torch.int8, device='cuda')


@torch.inference_mode()
def run_inference(factory, inp):
    impl = factory(inp.shape[-1], eps, False)
    impl.to(dtype=inp.dtype, device=inp.device)
    impl.eval()

    for _ in range(warmup_runs):
        impl(inp)

    events = make_events()
    cache = make_cache()

    for i in range(num_runs):
        cache.zero_()
        events[i][0].record()
        impl(inp)
        events[i][1].record()

    torch.cuda.synchronize()
    times = [[events[i][0].elapsed_time(events[i][1])] for i in range(num_runs)]
    return np.asarray(times)


def run_training(factory, inp):
    impl = factory(inp.shape[-1], eps, True)
    impl.to(dtype=inp.dtype, device=inp.device)
    impl.train()

    inp.requires_grad = True

    for _ in range(warmup_runs):
        out = impl(inp)
        out.sum().backward()

    inp.grad = None
    impl.weight.grad = None

    events = make_events()
    events2 = make_events()
    cache = make_cache()

    for i in range(num_runs):
        cache.zero_()
        events[i][0].record()
        out = impl(inp)
        events[i][1].record()

        out = out.sum()

        cache.zero_()
        events2[i][0].record()
        out.backward()
        events2[i][1].record()

        inp.grad = None
        impl.weight.grad = None

    torch.cuda.synchronize()

    times = [[events[i][0].elapsed_time(events[i][1]), events2[i][0].elapsed_time(events2[i][1])] for i in range(num_runs)]
    return np.asarray(times)


def run(factory, inp, training):
    if training:
        return run_training(factory, inp)
    else:
        return run_inference(factory, inp)


def fmt(m, s):
    return f'{m:.4f} ± {s:.4f}'


def summarise(times):
    mean = np.mean(times, axis=0)
    std = np.std(times, axis=0)
    return [fmt(m, s) for m, s in zip(mean, std)]


def main():
    impls = dict(I.get_impls())
    impls = { key: impls[key] for key in sorted(impls) }

    torch.random.manual_seed(12345)

    table = [['Shape', 'Type', 'Impl.', 'Inf.', 'Fwd.', 'Bwd.']]

    for shape, dtype in product(shapes, dtypes):
        if len(table) > 1:
            table.append(['', '', '', '', '', ''])

        with torch.no_grad():
            inp = torch.rand(shape, dtype=dtype, device='cuda') * 10

        for name, impl in impls.items():
            results = []
            for training in [False, True]:
                try:
                    out = run(impl, inp, training)
                    out = summarise(out)
                    results.extend(out)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    if not training:
                        results.append('Failed')
                    else:
                        results.extend(['Failed', 'Failed'])

            table.append([shape, dtype, name, *results])


    print(tabulate.tabulate(table, headers='firstrow'))


if __name__ == "__main__":
    main()
