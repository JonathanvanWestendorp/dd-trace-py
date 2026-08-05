from __future__ import annotations

from unittest import mock

import pytest

from ddtrace.internal.datadog.profiling import stack
from ddtrace.profiling import _asyncio as profiling_asyncio


pytestmark = pytest.mark.skipif(not stack.is_available, reason="stack profiler not available")


def test_failed_loop_registration_records_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    drops = []
    monkeypatch.setattr(profiling_asyncio.stack, "track_asyncio_loop", lambda thread_id, loop: False)
    monkeypatch.setattr(profiling_asyncio.stack, "_record_span_link_drop", lambda *args: drops.append(args))

    assert not profiling_asyncio._track_asyncio_loop(12, mock.sentinel.loop)
    assert drops == [("thread_not_registered", stack.SpanLinkDomain.ASYNCIO_TASK)]


def test_current_task_provider_requires_native_loop_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    task = mock.Mock()
    monkeypatch.setattr(profiling_asyncio, "get_running_loop", lambda: mock.sentinel.loop)
    monkeypatch.setattr(profiling_asyncio, "current_task", lambda: task)
    monkeypatch.setattr(profiling_asyncio, "_ensure_task_span_finalizer", lambda candidate: candidate is task)
    monkeypatch.setattr(profiling_asyncio.stack, "is_asyncio_loop_registered", lambda thread_id: False)

    assert profiling_asyncio._current_task_span_target() is None

    monkeypatch.setattr(profiling_asyncio.stack, "is_asyncio_loop_registered", lambda thread_id: True)
    assert profiling_asyncio._current_task_span_target() == stack.LogicalSpanTarget(
        stack.SpanLinkDomain.ASYNCIO_TASK, id(task)
    )


def test_completed_task_clears_mapping_before_object_finalization(monkeypatch: pytest.MonkeyPatch) -> None:
    cleared = []
    monkeypatch.setattr(profiling_asyncio, "_clear_native_task_span", lambda task_id: cleared.append(task_id))

    class Task:
        callback = None

        def add_done_callback(self, callback):
            self.callback = callback

    task = Task()
    assert profiling_asyncio._ensure_task_span_finalizer(task)
    assert task.callback is not None

    task.callback(task)

    assert cleared == [id(task)]
    assert id(task) not in profiling_asyncio._task_span_finalizers


@pytest.mark.subprocess()
def test_nested_creation_wrappers_publish_task_only_once() -> None:
    import asyncio as aio
    from unittest import mock

    from ddtrace.profiling import _asyncio as profiling_asyncio

    async def main():
        task = aio.current_task()
        assert task is not None
        profiling_asyncio._task_span_publications.discard(task)
        with mock.patch.object(profiling_asyncio.stack, "link_logical_span_context", return_value=False) as publish:
            profiling_asyncio._publish_task_span(task, None, False)
            profiling_asyncio._publish_task_span(task, None, False)
        publish.assert_called_once()

    aio.run(main())


def test_fork_reset_detaches_finalizers_and_publication_markers() -> None:
    class Task:
        def add_done_callback(self, callback):
            self.callback = callback

    task = Task()
    assert profiling_asyncio._ensure_task_span_finalizer(task)
    profiling_asyncio._task_span_publications.add(task)

    profiling_asyncio._reset_task_span_state_after_fork()

    assert not profiling_asyncio._task_span_finalizers
    assert not profiling_asyncio._task_span_publications


@pytest.mark.subprocess()
def test_completed_retained_tasks_do_not_accumulate_native_links() -> None:
    import asyncio as aio

    from ddtrace.internal.datadog.profiling import stack
    from ddtrace.profiling import profiler
    from ddtrace.trace import tracer

    retained = []

    async def child():
        await aio.sleep(0)

    async def main():
        with tracer.trace("long.request"):
            baseline = stack._stack.span_link_count(stack.SpanLinkDomain.ASYNCIO_TASK)
            retained.extend(aio.create_task(child()) for _ in range(100))
            await aio.gather(*retained)
            await aio.sleep(0)
            assert stack._stack.span_link_count(stack.SpanLinkDomain.ASYNCIO_TASK) == baseline

    p = profiler.Profiler(tracer=tracer)
    p.start()
    try:
        aio.run(main())
    finally:
        p.stop()
