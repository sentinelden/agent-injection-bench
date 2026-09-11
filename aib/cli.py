"""Command line interface.

    aib run      --target naive --defense none
    aib matrix   --targets naive,paranoid --defenses none,delimit,spotlight
    aib validate
    aib describe io-001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import adapters, dataset, defenses
from .adapters import reference  # noqa: F401  -- registers built-in targets
from .harness import Result, Summary, run, summarise

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "attacks.jsonl"


def _load(path: str | None) -> list[dataset.Attack]:
    return dataset.load(path or DEFAULT_CORPUS)


def _make_target(spec: str):
    """`name` or `name:key=value,key=value`."""
    if ":" not in spec:
        return adapters.get(spec)
    name, _, rest = spec.partition(":")
    kwargs = {}
    for pair in rest.split(","):
        if "=" in pair:
            key, _, value = pair.partition("=")
            kwargs[key.strip()] = value.strip()
    return adapters.get(name, **kwargs)


def _print_summary(summary: Summary, verbose: bool = False) -> None:
    print(f"\n  target:  {summary.target}")
    print(f"  defense: {summary.defense}")
    print(f"  {'─' * 56}")
    print(f"  compromise rate    {summary.compromise_rate:6.1%}   "
          f"({summary.compromised}/{summary.total_attacks - summary.errors} attacks)")
    print(f"  over-refusal rate  {summary.over_refusal_rate:6.1%}   "
          f"({summary.over_refused}/{summary.total_controls} controls)")
    if summary.errors:
        print(f"  errors             {summary.errors}")

    if verbose and summary.by_family:
        print(f"\n  by family")
        for family, (compromised, total) in sorted(summary.by_family.items(),
                                                   key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
            bar = "█" * round(10 * compromised / total) if total else ""
            print(f"    {family:<24} {compromised}/{total:<4} {bar}")
        print(f"\n  by channel")
        for channel, (compromised, total) in sorted(summary.by_channel.items(),
                                                    key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
            bar = "█" * round(10 * compromised / total) if total else ""
            print(f"    {channel:<24} {compromised}/{total:<4} {bar}")


def cmd_run(args) -> int:
    attacks = _load(args.corpus)
    target = _make_target(args.target)
    defense = defenses.get(args.defense)

    results: list[Result] = []

    def report(result: Result) -> None:
        if not args.quiet:
            mark = {"compromised": "FAIL", "resisted": "ok  ",
                    "over_refused": "OVER", "error": "ERR "}[result.outcome.value]
            print(f"  [{mark}] {result.attack_id:<10} {result.family:<22} {result.reason}")
        results.append(result)

    if not args.quiet:
        print(f"running {len(attacks)} attacks · target={args.target} · defense={args.defense}\n")
    run(attacks, target, defense, on_result=report)

    summary = summarise(results)
    if args.json:
        payload = {"summary": summary.to_dict(), "results": [r.to_dict() for r in results]}
        text = json.dumps(payload, indent=2)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"wrote {args.output}")
        else:
            print(text)
    else:
        _print_summary(summary, verbose=True)

    # Non-zero when anything was compromised, so a CI job can gate on it.
    return 1 if summary.compromised else 0


def cmd_matrix(args) -> int:
    attacks = _load(args.corpus)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    defense_names = [d.strip() for d in args.defenses.split(",") if d.strip()]

    rows: list[Summary] = []
    for target_spec in targets:
        target = _make_target(target_spec)
        for defense_name in defense_names:
            results = run(attacks, target, defenses.get(defense_name))
            rows.append(summarise(results))

    width = max((len(r.target) for r in rows), default=6)
    print(f"\n  {'target':<{width}}  {'defense':<12}  {'compromised':>11}  {'over-refused':>12}")
    print(f"  {'─' * (width + 42)}")
    for row in rows:
        print(f"  {row.target:<{width}}  {row.defense:<12}  "
              f"{row.compromise_rate:>10.1%}  {row.over_refusal_rate:>11.1%}")
    print()

    if args.json:
        text = json.dumps([r.to_dict() for r in rows], indent=2)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"wrote {args.output}")
        else:
            print(text)
    return 0


def cmd_validate(args) -> int:
    try:
        attacks = _load(args.corpus)
    except dataset.DatasetError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1

    families = dataset.by_family(attacks)
    channels = dataset.by_channel(attacks)
    controls = [a for a in attacks if a.is_control]

    print(f"  {len(attacks)} attacks, valid")
    print(f"  {len(attacks) - len(controls)} adversarial, {len(controls)} controls")
    print(f"\n  families ({len(families)})")
    for name, items in sorted(families.items()):
        print(f"    {name:<24} {len(items)}")
    print(f"\n  channels ({len(channels)})")
    for name, items in sorted(channels.items()):
        print(f"    {name:<24} {len(items)}")

    if not controls:
        print("\n  warning: no control cases; over-refusal cannot be measured",
              file=sys.stderr)
        return 1
    return 0


def cmd_describe(args) -> int:
    attacks = {a.id: a for a in _load(args.corpus)}
    attack = attacks.get(args.attack_id)
    if not attack:
        print(f"no attack with id {args.attack_id!r}", file=sys.stderr)
        return 1

    print(f"\n  {attack.id}  [{attack.family} / {attack.channel}]")
    print(f"  {attack.description}\n")
    print(f"  task:     {attack.task}")
    print(f"  injected: {attack.injected[:300]}")
    print(f"  success:  {attack.success.kind}")
    if attack.tools:
        print(f"  tools:    {', '.join(t.name for t in attack.tools)}")
    if attack.source:
        print(f"  source:   {attack.source}")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aib",
        description="Benchmark prompt injection against tool-calling agents.",
    )
    parser.add_argument("--corpus", help="path to a JSONL corpus (default: bundled)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the corpus against one target")
    p_run.add_argument("--target", default="naive",
                       help=f"target adapter; one of {', '.join(adapters.available())}, "
                            "or name:key=value for options")
    p_run.add_argument("--defense", default="none",
                       help=f"defense; one of {', '.join(defenses.available())}")
    p_run.add_argument("--json", action="store_true", help="emit machine-readable output")
    p_run.add_argument("--output", help="write JSON to this path")
    p_run.add_argument("--quiet", action="store_true", help="summary only")
    p_run.set_defaults(func=cmd_run)

    p_matrix = sub.add_parser("matrix", help="every target against every defense")
    p_matrix.add_argument("--targets", default="naive,paranoid")
    p_matrix.add_argument("--defenses", default="none,delimit,spotlight,filter,allowlist")
    p_matrix.add_argument("--json", action="store_true")
    p_matrix.add_argument("--output")
    p_matrix.set_defaults(func=cmd_matrix)

    p_validate = sub.add_parser("validate", help="check corpus integrity")
    p_validate.set_defaults(func=cmd_validate)

    p_describe = sub.add_parser("describe", help="show one attack in full")
    p_describe.add_argument("attack_id")
    p_describe.set_defaults(func=cmd_describe)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
