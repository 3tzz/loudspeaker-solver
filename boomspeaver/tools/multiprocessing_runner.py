# pylint: disable=missing-module-docstring

import multiprocessing
from multiprocessing import Manager, Pool
from typing import Any, Callable, Iterable

from rich.progress import Progress, TaskID


class MultiprocessingRunner:
    def __init__(
        self, function: Callable, args_list: Iterable[Any], num_workers: int = 4
    ):
        """
        General runner for multiprocessing.

        Arguments:
            function Callable to be executed in parallel.
            args_list: Iterable of arguments (passed one by one to the function).
            num_workers: Number of processes to run in parallel.
        """
        self.function = function
        self.args_list = list(args_list)
        self.num_workers = num_workers

    @staticmethod
    def get_cpu_num(verbose: bool = True) -> int:
        """Get number of cpu cores."""
        num_cores = multiprocessing.cpu_count()
        if verbose:
            print(f"Detected {num_cores} CPU cores.")
        return num_cores

    def _wrapped_function(self, args, progress_queue):
        """Call the original function and notify progress queue."""
        result = self.function(args)
        progress_queue.put(1)
        return result

    def run(self):
        """Run initialized earlier Callable with arguments."""
        with Manager() as manager:
            progress_queue = manager.Queue()

            with Progress() as progress:
                task_id = progress.add_task(
                    "[cyan]Processing...", total=len(self.args_list)
                )

                with Pool(processes=self.num_workers) as pool:
                    watcher = multiprocessing.Process(
                        target=self._progress_watcher,
                        args=(progress_queue, progress, task_id),
                    )
                    watcher.start()

                    pool.starmap(
                        self._wrapped_function,
                        [(args, progress_queue) for args in self.args_list],
                    )

                    progress_queue.put(None)
                    watcher.join()

    @staticmethod
    def _progress_watcher(queue, progress: Progress, task_id: TaskID):
        """Listen for progress updates and refresh the progress bar."""
        while True:
            item = queue.get()
            if item is None:
                break
            progress.advance(task_id)


if __name__ == "__main__":
    import time

    def dummy_task(x):
        """Testing method."""
        time.sleep(0.5)
        return x * x

    if __name__ == "__main__":
        data = list(range(10))

        runner = MultiprocessingRunner(
            function=dummy_task, args_list=data, num_workers=4
        )
        runner.run()
