"""Tests for profiler span-link lifecycle independent of execution integrations."""

from __future__ import annotations

import contextvars
import sys
from types import SimpleNamespace

import pytest

from ddtrace._trace.context import Context
from ddtrace.internal.datadog.profiling import context_meta
from ddtrace.internal.datadog.profiling import stack as stack_module
from ddtrace.profiling.collector import stack as stack_collector


pytestmark = pytest.mark.skipif(not stack_module.is_available, reason="stack profiler not available")


@pytest.fixture(autouse=True)
def restore_span_linking_state():
    enabled = stack_module._span_linking_enabled
    generation = stack_module._span_link_generation
    active_span_link = stack_module._active_span_link.get()
    providers = list(stack_module._logical_span_providers)
    stack_module._span_linking_enabled = False
    stack_module._set_active_span_link(None)
    stack_module._logical_span_providers.clear()
    stack_module._stack.reset_span_links()
    try:
        yield
    finally:
        stack_module._stack.reset_span_links()
        stack_module._span_linking_enabled = enabled
        stack_module._span_link_generation = generation
        stack_module._set_active_span_link(active_span_link)
        stack_module._logical_span_providers[:] = providers


def _target(domain: stack_module.SpanLinkDomain, identifier: int) -> stack_module.LogicalSpanTarget:
    return stack_module.LogicalSpanTarget(domain, identifier)


def test_context_deactivation_clears_physical_span_link(monkeypatch: pytest.MonkeyPatch) -> None:
    cleared = []
    monkeypatch.setattr(stack_module._stack, "clear_span", lambda: cleared.append(True))

    stack_module.enable_span_linking()
    stack_module.link_span(None)

    assert cleared == [True]


def test_span_activation_uses_highest_priority_logical_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    linked = []
    monkeypatch.setattr(stack_module._stack, "link_logical_span", lambda *args: linked.append(args))
    monkeypatch.setattr(stack_module._stack, "link_span", lambda *args: pytest.fail("unexpected thread fallback"))

    def low_priority_provider():
        return _target(stack_module.SpanLinkDomain.GEVENT_GREENLET, 11)

    def high_priority_provider():
        return _target(stack_module.SpanLinkDomain.ASYNCIO_TASK, 22)

    stack_module.register_logical_span_provider(low_priority_provider, priority=10)
    stack_module.register_logical_span_provider(high_priority_provider, priority=20)
    stack_module.enable_span_linking()

    stack_module.link_span(Context(trace_id=1, span_id=101))
    stack_module.unregister_logical_span_provider(high_priority_provider)
    stack_module.link_span(Context(trace_id=2, span_id=202))

    assert linked == [
        (stack_module.SpanLinkDomain.ASYNCIO_TASK, 22, 101, 101, None),
        (stack_module.SpanLinkDomain.GEVENT_GREENLET, 11, 202, 202, None),
    ]


def test_provider_failure_records_drop_and_uses_next_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    drops = []
    monkeypatch.setattr(
        stack_module, "_record_span_link_drop", lambda reason, domain=None: drops.append((reason, domain))
    )

    def broken_provider():
        raise RuntimeError("provider failed")

    expected = _target(stack_module.SpanLinkDomain.GEVENT_GREENLET, 31)
    stack_module.register_logical_span_provider(broken_provider, priority=20)
    stack_module.register_logical_span_provider(lambda: expected, priority=10)

    assert stack_module._current_logical_span_target() == expected
    assert drops == [("provider_error", None)]


def test_snapshot_reports_mapping_count_by_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = []
    monkeypatch.setattr(stack_module._stack, "span_link_count", lambda domain: int(domain) + 4)
    monkeypatch.setattr(
        stack_collector.telemetry_writer,
        "add_gauge_metric",
        lambda namespace, name, value, tags: metrics.append((namespace, name, value, tags)),
    )

    stack_collector.StackCollector.snapshot()

    assert [(name, value, tags) for _, name, value, tags in metrics] == [
        ("span_links", 4, (("domain", "thread"),)),
        ("span_links", 5, (("domain", "asyncio_task"),)),
        ("span_links", 6, (("domain", "gevent_greenlet"),)),
    ]


def test_logical_detachment_does_not_clear_thread_link(monkeypatch: pytest.MonkeyPatch) -> None:
    linked = []
    cleared = []
    monkeypatch.setattr(stack_module._stack, "link_logical_span", lambda *args: linked.append(args))
    monkeypatch.setattr(stack_module._stack, "clear_logical_span", lambda *args: cleared.append(args))
    monkeypatch.setattr(stack_module._stack, "clear_span", lambda: pytest.fail("unexpected thread clear"))

    stack_module.register_logical_span_provider(lambda: _target(stack_module.SpanLinkDomain.GEVENT_GREENLET, 33))
    stack_module.enable_span_linking()
    stack_module.link_span(Context(trace_id=1, span_id=303))
    stack_module.link_span(None)

    assert linked == [(stack_module.SpanLinkDomain.GEVENT_GREENLET, 33, 303, 303, None)]
    assert cleared == [(stack_module.SpanLinkDomain.GEVENT_GREENLET, 33)]


def test_finished_span_cleanup_is_one_atomic_native_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    unlinked = []
    monkeypatch.setattr(stack_module._stack, "unlink_finished_span", lambda span_id: unlinked.append(span_id))

    stack_module.enable_span_linking()
    stack_module.unlink_finished_span(SimpleNamespace(span_id=404))

    assert unlinked == [404]


def test_finishing_local_root_preserves_copied_active_logical_descendants() -> None:
    descendant = Context(trace_id=1, span_id=502)
    context_meta.attach_profiler_link(descendant, local_root_span_id=501, span_type="web")

    stack_module.enable_span_linking()
    stack_module.link_logical_span(stack_module.SpanLinkDomain.ASYNCIO_TASK, 51, descendant)
    stack_module.link_logical_span(stack_module.SpanLinkDomain.GEVENT_GREENLET, 52, descendant)
    stack_module.link_logical_span(stack_module.SpanLinkDomain.ASYNCIO_TASK, 51, Context(trace_id=2, span_id=601))

    assert stack_module._stack.unlink_finished_span(501) == 0
    assert stack_module._stack.span_link_count(stack_module.SpanLinkDomain.ASYNCIO_TASK) == 1
    assert stack_module._stack.span_link_count(stack_module.SpanLinkDomain.GEVENT_GREENLET) == 1

    assert stack_module._stack.unlink_finished_span(502) == 1
    assert stack_module._stack.span_link_count(stack_module.SpanLinkDomain.ASYNCIO_TASK) == 1
    assert stack_module._stack.span_link_count(stack_module.SpanLinkDomain.GEVENT_GREENLET) == 0


def test_inherited_context_seeds_logical_span_for_current_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    linked = []
    cleared = []
    drops = []
    monkeypatch.setattr(stack_module._stack, "link_span", lambda *args: None)
    monkeypatch.setattr(stack_module._stack, "link_logical_span", lambda *args: linked.append(args))
    monkeypatch.setattr(stack_module._stack, "clear_logical_span", lambda *args: cleared.append(args))
    monkeypatch.setattr(stack_module, "_record_span_link_drop", lambda *args: drops.append(args))

    stack_module.enable_span_linking()
    stack_module.link_span(Context(trace_id=1, span_id=701))
    inherited_context = contextvars.copy_context()
    assert stack_module.link_logical_span_context(stack_module.SpanLinkDomain.ASYNCIO_TASK, 71, inherited_context)

    stack_module.disable_span_linking()
    stack_module.enable_span_linking()
    assert not stack_module.link_logical_span_context(stack_module.SpanLinkDomain.ASYNCIO_TASK, 72, inherited_context)

    assert linked == [(stack_module.SpanLinkDomain.ASYNCIO_TASK, 71, 701, 701, None)]
    assert cleared == [(stack_module.SpanLinkDomain.ASYNCIO_TASK, 72)]
    assert drops == [("stale_generation", stack_module.SpanLinkDomain.ASYNCIO_TASK)]


def test_clear_logical_span_is_domain_scoped() -> None:
    stack_module.enable_span_linking()
    context = Context(trace_id=1, span_id=602)
    stack_module.link_logical_span(stack_module.SpanLinkDomain.ASYNCIO_TASK, 61, context)
    stack_module.link_logical_span(stack_module.SpanLinkDomain.GEVENT_GREENLET, 61, context)

    stack_module.clear_logical_span(stack_module.SpanLinkDomain.ASYNCIO_TASK, 61)

    assert stack_module._stack.span_link_count(stack_module.SpanLinkDomain.ASYNCIO_TASK) == 0
    assert stack_module._stack.span_link_count(stack_module.SpanLinkDomain.GEVENT_GREENLET) == 1


def test_postfork_reset_invalidates_all_inherited_span_link_state(monkeypatch: pytest.MonkeyPatch) -> None:
    resets = []
    monkeypatch.setattr(stack_module._stack, "reset_span_links", lambda: resets.append(True))

    stack_module.enable_span_linking()
    stack_module.link_span(Context(trace_id=1, span_id=701))
    generation = stack_module._span_link_generation
    stack_module._reset_span_link_state()

    assert resets == [True, True]
    assert stack_module._span_link_generation == generation + 1
    assert stack_module._active_span_link.get() is None


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="safe ContextVar setter is only needed before Python 3.12")
def test_active_span_link_uses_safe_contextvar_set(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(stack_module, "safe_contextvar_set", lambda variable, value: calls.append((variable, value)))
    value = stack_module._SpanLinkContext(1, stack_module._SpanInfo(2, 3, "web"), None)

    stack_module._set_active_span_link(value)

    assert calls == [(stack_module._active_span_link, value)]
