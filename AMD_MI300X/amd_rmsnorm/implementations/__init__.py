
impls = {}


def implementation(name):
    def impl(fn):
        if name in impls:
            raise KeyError(f'Implementation with name {name!r} already exists')
        impls[name] = fn
        return fn

    return impl


def load_all():
    import importlib
    import os

    this_dir = os.path.dirname(__file__)
    for name in os.listdir(this_dir):
        path = os.path.join(this_dir, name)
        if path == __file__:
            continue

        to_load = None
        if name.endswith('.py'):
            to_load = name[:-3]
        elif os.path.isdir(path) and os.path.exists(os.path.join(path, '__init__.py')):
            to_load = name

        if to_load is not None:
            try:
                importlib.import_module('.' + to_load, package=__package__)
            except Exception as e:
                print(f'WARNING: Failed to load an implementation from: {os.path.join(this_dir, path)}: {e}')
                # import traceback
                # traceback.print_exc()
                continue


def get_impls():
    yield from impls.items()
