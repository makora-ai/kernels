from itertools import product

import torch
import tabulate

import implementations as I

I.load_all()

dtypes = [torch.float, torch.float16, torch.bfloat16]
shapes = [(1, 4096, 8192)]
eps = 1e-6

warmup_runs = 3


@torch.inference_mode()
def run_inference(factory, inp):
    impl = factory(inp.shape[-1], eps, False)
    impl.to(dtype=inp.dtype, device=inp.device)
    impl.eval()

    for _ in range(warmup_runs):
        impl(inp)

    out = impl(inp)
    torch.cuda.synchronize()
    return out


def run_training(factory, inp):
    impl = factory(inp.shape[-1], eps, True)
    impl.to(dtype=inp.dtype, device=inp.device)
    impl.train()

    for _ in range(warmup_runs):
        out = impl(inp)
        out.sum().backward()

    impl.weight.grad = None

    out = impl(inp)
    out.sum().backward()

    torch.cuda.synchronize()

    return impl.weight.grad


def run(factory, inp, training):
    if training:
        return run_training(factory, inp)
    else:
        return run_inference(factory, inp)


@torch.inference_mode()
def compare(out, ref):
    atol = 1e-8 if out.dtype == torch.float32 else 1e-6
    rtol = 1e-5 if out.dtype == torch.float32 else 1e-2
    if not torch.allclose(out, ref, rtol=rtol, atol=atol):
        err = (out - ref).abs() / (ref.abs() + eps)
        mean = err.mean()
        max = err.max()
        min = err.min()
        return f'Not close: {float(max)} -- {float(mean)} -- {float(min)}'

    return 'OK'


def main():
    impls = dict(I.get_impls())
    impls = { key: impls[key] for key in sorted(impls) }
    ref_impl = impls.pop('pytorch')

    torch.random.manual_seed(12345)

    table = [['Shape', 'Type', 'Impl.', 'Fwd.', 'Bwd.']]

    for shape, dtype in product(shapes, dtypes):
        if len(table) > 1:
            table.append(['', '', '', '', ''])

        with torch.no_grad():
            inp = torch.rand(shape, dtype=dtype, device='cuda') * 10
        
        for name, impl in impls.items():
            results = []
            for training in [False, True]:
                ref = run(ref_impl, inp, training)
                try:
                    out = run(impl, inp, training)
                    res = compare(out, ref)
                    results.append(res)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    results.append(f'Error: {e}')

            table.append([str(shape), str(dtype), name, *results])

    print(tabulate.tabulate(table, headers='firstrow'))


if __name__ == '__main__':
    main()
