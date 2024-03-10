"""Basic parallel processes."""

import multiprocessing
import tqdm
import tqdm.auto

from joblib import Parallel, delayed


class ProgressParallel(Parallel):

    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm.auto.tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def parallelize(array, function, n_jobs=None, use_list_args=False,
                use_kwargs=False, use_tqdm=True):
    assert use_list_args is False or use_kwargs is False
    if use_list_args:
        def wrapped_function(x):
            return function(*x)
    elif use_kwargs:
        def wrapped_function(x):
            return function(**x)
    else:
        def wrapped_function(x):
            return function(x)
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()-1
    # If we set n_jobs to 1, just run a list comprehension.
    # This is useful for benchmarking and debugging among other things.
    if n_jobs==1:
        if use_tqdm:
            return [wrapped_function(a) for a in tqdm.auto.tqdm(array)]
        else:
            return [wrapped_function(a) for a in array]

    pp = ProgressParallel(n_jobs=n_jobs, total=len(array), use_tqdm=use_tqdm)

    return pp(delayed(wrapped_function)(a) for a in array)
