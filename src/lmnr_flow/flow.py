import logging
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from inspect import signature
from queue import Queue
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from lmnr import Laminar, observe

from .context import Context
from .state import State

__ERROR__ = "__ERROR__"
__OUTPUT__ = "__OUTPUT__"
__HASH_SPLIT__ = "____"


@dataclass
class NextTask:
    """
    Represents a task that is scheduled to run next in the flow.

    Attributes:
        id (str): The unique identifier of the next task.
        inputs (Optional[Dict[str, Any]]): A dictionary of inputs to be passed to the next task. Defaults to None.
        spawn_another (bool): If true, task will be executed even though there is already an instance of the same task running. This is useful when you want to run the same task in parallel. Defaults to False.
    """

    id: str
    inputs: Optional[Dict[str, Any]] = None
    spawn_another: bool = False


@dataclass
class TaskOutput:
    output: Any
    next_tasks: Optional[List[NextTask]] = None


@dataclass
class Task:
    id: str
    action: Callable[[Context], TaskOutput]


@dataclass
class StreamChunk:
    task_id: str
    value: Any


class Flow:
    def __init__(
        self,
        thread_pool_executor: ThreadPoolExecutor,
        context: Optional[Context] = None,
    ):
        self.tasks = {}  # str -> Task
        self.active_tasks = set()  # Set of str
        self.context = context or Context()  # Global context
        self.output_task_ids = set()  # Set of str
        self._executor = thread_pool_executor

        # Thread-safety locks
        self.active_tasks_lock = Lock()
        self.output_ids_lock = Lock()
        self.logger = logging.getLogger(__name__)

    def add_task(self, name: str, action: Callable[[Context], TaskOutput]):
        self.context.set_state(name, State.empty())
        self.tasks[name] = Task(name, action)
        self.logger.info(f"Added task '{name}'")

    def task(
        self, name: str
    ) -> Callable[[Callable[[Context], TaskOutput]], Callable[[Context], TaskOutput]]:
        """
        Decorator to register a task in the flow.

        Args:
            name: Name for the task.

        Returns:
            Decorated function.

        Example:
            @flow.task("process_data")
            def process_data(ctx):
                return TaskOutput(output=process(data))
        """

        def decorator(
            func: Callable[[Context], TaskOutput],
        ) -> Callable[[Context], TaskOutput]:
            self.add_task(name, func)
            return func

        return decorator

    def execute_task(
        self,
        action: Callable[[Context], TaskOutput],
        task: NextTask,
        task_queue: Queue,
        stream_queue: Optional[Queue] = None,
    ):
        self.logger.info(f"Starting execution of task '{task.id}'")

        try:
            with Laminar.start_as_current_span(
                task.id,
                input={"context": self.context.to_dict(), "inputs": task.inputs},
            ):
                # Check if action accepts inputs parameter
                sig = signature(action)
                if "inputs" in sig.parameters:
                    result: TaskOutput = action(self.context, inputs=task.inputs)
                else:
                    result: TaskOutput = action(self.context)
                Laminar.set_span_output(result)

            # Set state to the output of the task
            self.context.set(task.id, result.output)

            # Push to stream queue if it exists
            if stream_queue is not None:
                stream_queue.put(StreamChunk(task.id, result.output))

            with self.active_tasks_lock:
                self.active_tasks.remove(task.id)
                self.logger.info(f"Completed execution of task '{task.id}'")

                # If no next tasks specified, this is an output task
                if not result.next_tasks or len(result.next_tasks) == 0:
                    self.logger.info(f"Task '{task.id}' completed as output node")
                    with self.output_ids_lock:
                        self.output_task_ids.add(task.id)
                        task_queue.put(NextTask(__OUTPUT__, None))
                else:
                    self.logger.debug(
                        f"Task '{task.id}' scheduling next tasks: {result.next_tasks}"
                    )

                    for next_task in result.next_tasks:
                        if next_task.id.split(__HASH_SPLIT__)[0] in self.tasks:
                            if next_task.id not in self.active_tasks:
                                self.active_tasks.add(next_task.id)
                                task_queue.put(NextTask(next_task.id, next_task.inputs))
                            elif next_task.spawn_another:
                                self.logger.info(
                                    f"Spawning another instance of task '{next_task.id}'"
                                )
                                task_id_with_hash = (
                                    next_task.id
                                    + __HASH_SPLIT__
                                    + str(uuid.uuid4())[0:8]
                                )
                                self.active_tasks.add(task_id_with_hash)
                                task_queue.put(
                                    NextTask(task_id_with_hash, next_task.inputs)
                                )
                        else:
                            raise Exception(f"Task {next_task.id} not found")

        except Exception as e:
            self.context.set(
                __ERROR__, {"error": str(e), "traceback": traceback.format_exc()}
            )
            with self.active_tasks_lock:
                self.active_tasks.clear()

            task_queue.put(NextTask(__ERROR__, None))

            raise e

    @observe(name="flow.run")
    def run(
        self, start_task_id: str, inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        self.logger.info(f"Starting engine run with initial task: {start_task_id}")
        # thread-safe queue of task ids
        task_queue = Queue()
        futures = set()

        self.active_tasks.add(start_task_id)
        task_queue.put(NextTask(start_task_id, inputs))

        if inputs:
            for key, value in inputs.items():
                self.context.set(key, value)

        # Main execution loop
        while True:
            next_task = task_queue.get()

            if next_task.id == __ERROR__:
                # Cancel all pending futures on error
                for f in futures:
                    f.cancel()

                err = self.context.get(__ERROR__)
                raise Exception(err)

            if next_task.id == __OUTPUT__:
                with self.active_tasks_lock:
                    if len(self.active_tasks) == 0:
                        break
                continue

            action = self.tasks[next_task.id.split(__HASH_SPLIT__)[0]].action

            future = self._executor.submit(
                self.execute_task, action, next_task, task_queue
            )
            futures.add(future)

        # Return values of the output nodes
        # task_id -> value of the task
        return {task_id: self.context.get(task_id) for task_id in self.output_task_ids}

    @observe(name="flow.stream")
    def stream(self, start_task_id: str, inputs: Optional[Dict[str, Any]] = None):
        task_queue = Queue()
        stream_queue = Queue()
        futures = set()

        self.active_tasks.add(start_task_id)
        task_queue.put(NextTask(start_task_id, inputs))

        if inputs:
            for key, value in inputs.items():
                self.context.set(key, value)

        self.context.set_stream(stream_queue)

        def run_engine():
            while True:
                next_task = task_queue.get()

                if next_task.id == __ERROR__:
                    for f in futures:
                        f.cancel()
                    stream_queue.put(StreamChunk(__ERROR__, None))  # Signal completion
                    break

                if next_task.id == __OUTPUT__:
                    with self.active_tasks_lock:
                        if len(self.active_tasks) == 0:
                            stream_queue.put(
                                StreamChunk(__OUTPUT__, None)
                            )  # Signal completion
                            break
                    continue

                action = self.tasks[next_task.id.split(__HASH_SPLIT__)[0]].action

                future = self._executor.submit(
                    self.execute_task, action, next_task, task_queue, stream_queue
                )
                futures.add(future)

        self._executor.submit(run_engine)

        # Yield results from stream queue
        while True:
            stream_chunk: StreamChunk = stream_queue.get()
            if (
                stream_chunk.task_id == __OUTPUT__ or stream_chunk.task_id == __ERROR__
            ):  # Check for completion signal
                break
            yield stream_chunk

    def get_context(self) -> Context:
        return self.context
