#!/usr/bin/env python3
"""Fail-closed max-plus replay for Colibri event traces (stdlib only)."""

import argparse
import json
import sys
from collections import deque
from pathlib import Path


def fail(message):
    raise ValueError(message)


def load_json(path):
    with path.open() as source:
        return json.load(source)


def load_events(path, token_position):
    """Load a normalized graph, or the safe outer-layer graph from result JSONL."""
    if path.suffix == ".json":
        graph = load_json(path)
        if graph.get("schema") != "colibri-maxplus-events-v1":
            fail("normalized graph schema must be colibri-maxplus-events-v1")
        events = graph.get("events")
        if not isinstance(events, list):
            fail("normalized graph events must be a list")
        return events, graph.get("observed_makespan_ns")

    records = [json.loads(line) for line in path.open()]
    layers = sorted(
        (record for record in records
         if record.get("event") == "layer" and record.get("token_position") == token_position),
        key=lambda record: record["layer"],
    )
    token = next(
        (record for record in records
         if record.get("event") == "token" and record.get("token_position") == token_position),
        None,
    )
    if not layers or token is None:
        fail("result JSONL needs layer records and one matching token record")
    events = []
    for index, record in enumerate(layers):
        if record.get("layer") != index or not isinstance(record.get("layer_elapsed_ns"), int):
            fail("result JSONL needs contiguous layers with integer layer_elapsed_ns")
        events.append({
            "id": "layer:%d:%d" % (token_position, index),
            "duration_ns": record["layer_elapsed_ns"],
            "kind": "exclusive",
            "deps": [] if index == 0 else ["layer:%d:%d" % (token_position, index - 1)],
            "resource": "layer-transaction",
        })
    return events, token["e2e_ns"]


def replay(events, replacements):
    by_id = {}
    for event in events:
        event_id = event.get("id")
        kind = event.get("kind")
        duration = event.get("duration_ns")
        deps = event.get("deps", [])
        if not isinstance(event_id, str) or event_id in by_id:
            fail("events need unique string ids")
        if kind not in ("exclusive", "opaque"):
            fail("only exclusive or opaque events are replayable; inclusive timing is rejected")
        if not isinstance(duration, int) or duration < 0 or not isinstance(deps, list):
            fail("events need non-negative integer duration_ns and deps list")
        by_id[event_id] = dict(event)
    for event_id, duration in replacements.items():
        if event_id not in by_id or not isinstance(duration, int) or duration < 0:
            fail("counterfactual replacement must name an event and use a non-negative integer")
        by_id[event_id]["duration_ns"] = duration

    children = {event_id: [] for event_id in by_id}
    indegree = {event_id: 0 for event_id in by_id}
    for event_id, event in by_id.items():
        for dependency in event["deps"]:
            if dependency not in by_id:
                fail("event %s depends on unknown %s" % (event_id, dependency))
            children[dependency].append(event_id)
            indegree[event_id] += 1
    ready = deque(event_id for event_id, count in indegree.items() if count == 0)
    finish, predecessor = {}, {}
    while ready:
        event_id = ready.popleft()
        event = by_id[event_id]
        deps = event["deps"]
        if deps:
            parent = max(deps, key=finish.__getitem__)
            predecessor[event_id] = parent
            start = finish[parent]
        else:
            start = 0
        finish[event_id] = start + event["duration_ns"]
        for child in children[event_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(finish) != len(by_id):
        fail("event graph contains a cycle")
    tail = max(finish, key=finish.__getitem__)
    critical_path = []
    while True:
        critical_path.append(tail)
        if tail not in predecessor:
            break
        tail = predecessor[tail]
    critical_path.reverse()
    work = {}
    for event in by_id.values():
        resource = event.get("resource")
        if resource and event["kind"] == "exclusive":
            work[resource] = work.get(resource, 0) + event["duration_ns"]
    return finish[max(finish, key=finish.__getitem__)], critical_path, work


def run_self_test():
    makespan, path, _ = replay([
        {"id": "a", "duration_ns": 5, "kind": "exclusive", "deps": []},
        {"id": "b", "duration_ns": 7, "kind": "opaque", "deps": ["a"]},
    ], {})
    if (makespan, path) != (12, ["a", "b"]):
        fail("self-test max-plus chain failed")
    try:
        replay([{"id": "bad", "duration_ns": 1, "kind": "inclusive", "deps": []}], {})
    except ValueError:
        return
    fail("self-test accepted inclusive timing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, nargs="?")
    parser.add_argument("--token-position", type=int, default=1)
    parser.add_argument("--counterfactual", type=Path)
    parser.add_argument("--max-relative-error", type=float)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        print('{"status":"PASS","self_test":"max-plus and inclusive rejection"}')
        return
    if args.trace is None:
        parser.error("trace is required unless --self-test is used")
    replacements = {}
    if args.counterfactual:
        counterfactual = load_json(args.counterfactual)
        replacements = counterfactual.get("replace_duration_ns", {})
        if set(counterfactual) != {"replace_duration_ns"} or not isinstance(replacements, dict):
            fail("counterfactual must contain only replace_duration_ns")
    events, observed = load_events(args.trace, args.token_position)
    makespan, critical_path, work = replay(events, replacements)
    error = None if observed is None else observed - makespan
    relative_error = None if observed in (None, 0) else abs(error) / observed
    if args.max_relative_error is not None and (relative_error is None or relative_error > args.max_relative_error):
        fail("replay relative error exceeds --max-relative-error")
    print(json.dumps({
        "observed_makespan_ns": observed,
        "replay_makespan_ns": makespan,
        "error_ns": error,
        "relative_error": relative_error,
        "critical_path": critical_path,
        "resource_exclusive_work_ns": work,
        "changed_nodes": sorted(replacements),
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print("FAIL: %s" % error, file=sys.stderr)
        sys.exit(2)
