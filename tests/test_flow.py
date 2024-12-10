import logging
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from lmnr.openllmetry_sdk.tracing.tracing import TracerWrapper
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from src.lmnr_flow.flow import Context, Flow, NextTask, StreamChunk, TaskOutput

# Configure logging with thread information
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    force=True,
)


@pytest.fixture(autouse=True)
def setup_tracer():
    # Create and set the tracer provider
    tracer_provider = TracerProvider()
    # Optional: Add console exporter to see the traces in the console
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    # Set the tracer provider for the wrapper
    wrapper = TracerWrapper()
    wrapper.tracer_provider = tracer_provider

    yield

    # Cleanup after tests
    wrapper.tracer_provider = None


@pytest.fixture(autouse=True)
def mock_tracer():
    with patch("lmnr.openllmetry_sdk.tracing.tracing.TracerWrapper.get_tracer"):
        yield


@pytest.fixture(autouse=True)
def mock_start_as_current_span():
    class MockContextManager:
        def __init__(self, func, input=None):
            self.func = func

        def __enter__(self):
            return self.func

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("lmnr.Laminar.start_as_current_span", side_effect=MockContextManager):
        yield


@pytest.fixture
def thread_pool():
    with ThreadPoolExecutor(max_workers=2) as executor:
        yield executor


@pytest.fixture
def flow(thread_pool):
    return Flow(thread_pool)


@pytest.fixture
def flow_with_state(thread_pool):
    context = Context()
    context.from_dict({"task1": "result1"})

    return Flow(thread_pool, context)


def test_simple_task_execution(flow):
    # Test single task that returns no next tasks
    def action(ctx):
        return TaskOutput("result")

    flow.add_task("task1", action)
    result = flow.run("task1")

    assert result == {"task1": "result"}
    assert flow.context.get("task1") == "result"


def test_sequential_tasks(flow):
    # Test chain of tasks
    def task1(ctx):
        return TaskOutput("result1", [NextTask("task2")])

    def task2(ctx):
        assert ctx.get("task1") == "result1"
        return TaskOutput("result2")

    flow.add_task("task1", task1)
    flow.add_task("task2", task2)

    result = flow.run("task1")
    assert result == {"task2": "result2"}


def test_parallel_tasks(flow):
    # Test tasks running in parallel
    def task1(ctx):
        return TaskOutput("result1", [NextTask("task2"), NextTask("task3")])

    def task2(ctx):
        return TaskOutput("result2")

    def task3(ctx):
        return TaskOutput("result3")

    flow.add_task("task1", task1)
    flow.add_task("task2", task2)
    flow.add_task("task3", task3)

    result = flow.run("task1")
    assert result == {"task2": "result2", "task3": "result3"}


def test_error_handling(flow):
    # Test error propagation
    def failing_task(ctx):
        raise ValueError("Task failed")

    flow.add_task("failing", failing_task)

    with pytest.raises(Exception) as exc_info:
        flow.run("failing")

    assert "Task failed" in str(exc_info.value)


def test_streaming(flow):
    # Test streaming functionality
    def task1(ctx):
        return TaskOutput("result1", [NextTask("task2")])

    def task2(ctx):
        return TaskOutput("result2")

    flow.add_task("task1", task1)
    flow.add_task("task2", task2)

    results = []
    for stream_chunk in flow.stream("task1"):
        results.append((stream_chunk.task_id, stream_chunk.value))

    assert len(results) == 2
    assert ("task1", "result1") in results
    assert ("task2", "result2") in results


def test_streaming_with_inputs(flow):
    def task1(ctx, inputs):
        assert inputs == {"input1": "value1"}
        return TaskOutput("result1", [NextTask("task2", inputs={"input1": "value1"})])

    def task2(ctx, inputs):
        assert inputs == {"input1": "value1"}
        return TaskOutput("result2")

    flow.add_task("task1", task1)
    flow.add_task("task2", task2)

    results = []
    for stream_chunk in flow.stream("task1", inputs={"input1": "value1"}):
        results.append((stream_chunk.task_id, stream_chunk.value))

    assert len(results) == 2
    assert ("task1", "result1") in results
    assert ("task2", "result2") in results


def test_streaming_within_task(flow):
    def task1(ctx):
        for i in range(3):
            ctx.get_stream().put(StreamChunk("task1", i))
        return TaskOutput("result1")

    flow.add_task("task1", task1)

    results = []

    for stream_chunk in flow.stream("task1"):
        results.append((stream_chunk.task_id, stream_chunk.value))

    assert len(results) == 4
    assert ("task1", 0) in results
    assert ("task1", 1) in results
    assert ("task1", 2) in results
    assert ("task1", "result1") in results


def test_context_sharing(flow):
    # Test context sharing between tasks
    def task1(ctx):
        ctx.set("shared", "shared_value")
        return TaskOutput("result1", [NextTask("task2")])

    def task2(ctx):
        assert ctx.get("shared") == "shared_value"
        return TaskOutput("result2")

    flow.add_task("task1", task1)
    flow.add_task("task2", task2)

    result = flow.run("task1")
    assert result == {"task2": "result2"}


def test_context_get_with_fallback(flow):
    # Tests context handling of existing and missing keys (with/without default).
    def existing_key_task(ctx):
        ctx.set("existing_key", "existing_value")
        return TaskOutput("output_with_key")

    flow.add_task("existing_key_task", existing_key_task)

    flow.run("existing_key_task")

    # Test retrieving an existing key.
    existing_value = flow.context.get("existing_key")
    assert existing_value == "existing_value"

    with pytest.raises(Exception) as exc_info:
        flow.context.get("non_existent_key")
    assert "Key non_existent_key not found in context" in str(exc_info.value)

    fallback_value = flow.context.get(
        "non_existent_key_with_default", default="fallback_value"
    )
    assert fallback_value == "fallback_value"


def test_invalid_task_reference(flow):
    # Test referencing non-existent task
    def task1(ctx):
        return TaskOutput("result1", [NextTask("non_existent_task")])

    flow.add_task("task1", task1)

    with pytest.raises(Exception) as exc_info:
        flow.run("task1")

    assert "Task non_existent_task not found" in str(exc_info.value)


def test_actual_parallel_execution(flow):
    # Test that tasks actually run in parallel by checking execution time
    def task1(ctx):
        return TaskOutput("result1", [NextTask("slow_task1"), NextTask("slow_task2")])

    def slow_task1(ctx):
        time.sleep(0.5)  # Sleep for 500ms
        return TaskOutput("slow_result1", None)

    def slow_task2(ctx):
        time.sleep(0.5)  # Sleep for 500ms
        return TaskOutput("slow_result2", None)

    flow.add_task("task1", task1)
    flow.add_task("slow_task1", slow_task1)
    flow.add_task("slow_task2", slow_task2)

    start_time = time.time()
    result = flow.run("task1")
    execution_time = time.time() - start_time

    # If tasks run sequentially, it would take > 1 second
    # If parallel, it should take ~0.5 seconds (plus small overhead)
    assert execution_time < 0.8  # Allow some overhead but ensure parallel execution
    assert result == {"slow_task1": "slow_result1", "slow_task2": "slow_result2"}


def test_run_with_inputs(flow):
    # Test running tasks with initial inputs
    def task1(ctx):
        assert ctx.get("input1") == "value1"
        assert ctx.get("input2") == "value2"
        return TaskOutput("result1", [NextTask("task2")])

    def task2(ctx):
        # Verify inputs are still accessible in subsequent tasks
        assert ctx.get("input1") == "value1"
        assert ctx.get("input2") == "value2"
        return TaskOutput("result2")

    flow.add_task("task1", task1)
    flow.add_task("task2", task2)

    inputs = {"input1": "value1", "input2": "value2"}

    result = flow.run("task1", inputs=inputs)
    assert result == {"task2": "result2"}


def test_correct_order_of_execution(flow):
    # Test that tasks are executed in the correct order
    def task1(ctx):
        return TaskOutput("result1", [NextTask("task2"), NextTask("task3")])

    def task2(ctx):
        t1 = ctx.get("task1")
        return TaskOutput(t1 + "result2", [NextTask("task4")])

    def task3(ctx):
        t1 = ctx.get("task1")
        return TaskOutput(t1 + "result3", [NextTask("task4")])

    def task4(ctx):
        t2 = ctx.get("task2")
        t3 = ctx.get("task3")

        return TaskOutput(t2 + t3)

    flow.add_task("task1", task1)
    flow.add_task("task2", task2)
    flow.add_task("task3", task3)
    flow.add_task("task4", task4)

    result = flow.run("task1")
    assert result == {"task4": "result1result2result1result3"}


def test_cycle(flow):
    # Test that cycles are detected and handled
    def task1(ctx):
        c = ctx.get("count")

        if c == 3:
            return TaskOutput("result1", [NextTask("task3")])

        return TaskOutput("result1", [NextTask("task2")])

    def task2(ctx):
        c = ctx.get("count")
        ctx.set("count", c + 1)
        return TaskOutput("result2", [NextTask("task1")])

    def task3(ctx):
        return TaskOutput("final")

    flow.add_task("task1", task1)
    flow.add_task("task2", task2)
    flow.add_task("task3", task3)

    result = flow.run("task1", inputs={"count": 0})
    assert result == {"task3": "final"}


def test_state_loading(flow_with_state):
    # Test that state is loaded correctly
    flow_with_state.add_task("task2", lambda ctx: TaskOutput("result2"))
    flow_with_state.run("task2")

    assert flow_with_state.context.get("task1") == "result1"
    assert flow_with_state.context.get("task2") == "result2"


def test_inputs_to_next_tasks(flow):
    # Test that inputs are passed to next tasks
    def task1(ctx):
        return TaskOutput("result1", [NextTask("task2", inputs={"input1": "value1"})])

    def task2(ctx, inputs):
        assert inputs["input1"] == "value1"
        return TaskOutput("result2")

    flow.add_task("task1", task1)
    flow.add_task("task2", task2)

    result = flow.run("task1")
    assert result == {"task2": "result2"}


def test_inputs_to_next_tasks_with_no_inputs(flow):
    # Test that inputs are passed to next tasks
    def task1(ctx):
        return TaskOutput("result1", [NextTask("task2")])

    def task2(ctx, inputs):
        assert inputs is None
        return TaskOutput("result2")

    flow.add_task("task1", task1)
    flow.add_task("task2", task2)

    result = flow.run("task1")
    assert result == {"task2": "result2"}


def test_inputs_to_first_task(flow):
    # Test that inputs are passed to the first task
    def task1(ctx, inputs):
        assert inputs == {"input1": "value1"}
        return TaskOutput("result1")

    flow.add_task("task1", task1)
    result = flow.run("task1", inputs={"input1": "value1"})
    assert result == {"task1": "result1"}


def test_push_the_same_task_multiple_times(flow):
    # Test that the same task can be pushed multiple times
    def task1(ctx):
        print("Executing task1")
        c = ctx.get("count") + 1
        ctx.set("count", c)

        if c == 3:
            return TaskOutput("result1")

        return TaskOutput("result1", [NextTask("task1")])

    flow.add_task("task1", task1)

    result = flow.run("task1", inputs={"count": 0})
    count = flow.get_context().get("count")
    assert count == 3
    assert result == {"task1": "result1"}


def test_map_reduce(flow):
    # Test map reduce functionality
    def task1(ctx):
        ctx.set("collector", [])

        return TaskOutput(
            "result1",
            [
                NextTask("task2", spawn_another=True),
                NextTask("task2", spawn_another=True),
                NextTask("task2", spawn_another=True),
            ],
        )

    def task2(ctx):
        collector = ctx.get("collector")
        collector.append("result2")
        ctx.set("collector", collector)

        return TaskOutput("", [NextTask("task3")])

    def task3(ctx):
        collector = ctx.get("collector")
        return TaskOutput(collector)

    flow.add_task("task1", task1)
    flow.add_task("task2", task2)
    flow.add_task("task3", task3)

    result = flow.run("task1")
    assert result == {"task3": ["result2", "result2", "result2"]}


def test_task_decorator(flow):
    # Test task decorator
    @flow.task("decorated_task")
    def action(ctx) -> TaskOutput:
        return TaskOutput("result")

    result = flow.run("decorated_task")
    assert result == {"decorated_task": "result"}
    assert flow.context.get("decorated_task") == "result"


def test_task_decorator_with_inputs(flow):
    # Test task decorator with inputs
    @flow.task("decorated_task")
    def action(ctx, inputs) -> TaskOutput:
        assert inputs == {"input1": "value1"}
        return TaskOutput("result")

    result = flow.run("decorated_task", inputs={"input1": "value1"})
    assert result == {"decorated_task": "result"}


def test_task_decorator_with_next_tasks(flow):
    # Test task decorator with next tasks
    @flow.task("task1")
    def task1(ctx) -> TaskOutput:
        return TaskOutput("result1", [NextTask("task2")])

    @flow.task("task2")
    def task2(ctx) -> TaskOutput:
        assert ctx.get("task1") == "result1"
        return TaskOutput("result2")

    result = flow.run("task1")
    assert result == {"task2": "result2"}
    assert flow.context.get("task1") == "result1"
    assert flow.context.get("task2") == "result2"


def test_task_decorator_error_handling(flow):
    # Test error handling in decorated task
    @flow.task("failing_task")
    def failing_task(ctx) -> TaskOutput:
        raise ValueError("Task failed")

    with pytest.raises(Exception) as exc_info:
        flow.run("failing_task")

    assert "Task failed" in str(exc_info.value)


def test_task_decorator_streaming(flow):
    # Test streaming with decorated task
    @flow.task("task1")
    def task1(ctx) -> TaskOutput:
        for i in range(3):
            ctx.get_stream().put(StreamChunk("task1", i))
        return TaskOutput("result1")

    results = []
    for stream_chunk in flow.stream("task1"):
        results.append((stream_chunk.task_id, stream_chunk.value))

    assert len(results) == 4  # 3 stream chunks + final result
    assert ("task1", 0) in results
    assert ("task1", 1) in results
    assert ("task1", 2) in results
    assert ("task1", "result1") in results


def test_task_error_propagation(flow):
    import logging

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    @flow.task("task1")
    def task1(ctx) -> TaskOutput:
        logger.debug("Executing task1")
        import threading

        logger.debug(f"task1 running in thread: {threading.current_thread().name}")
        result = TaskOutput("result1", [NextTask("task2")])
        logger.debug("task1 completed")
        return result

    @flow.task("task2")
    def task2(ctx) -> TaskOutput:
        logger.debug("Executing task2")
        import threading

        logger.debug(f"task2 running in thread: {threading.current_thread().name}")

        raise ValueError("Task2 failed")

    @flow.task("task3")
    def task3(ctx) -> TaskOutput:
        logger.debug("Executing task3")
        return TaskOutput("result3")  # Should not be executed

    with pytest.raises(Exception) as exc_info:
        flow.run("task1")

    assert "Task2 failed" in str(exc_info.value)
    assert flow.context.get("task1") == "result1"
