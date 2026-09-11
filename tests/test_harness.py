"""End-to-end behaviour of the harness and the summary arithmetic."""

from aib import defenses
from aib.adapters import get as get_target
from aib.adapters import reference  # noqa: F401 -- registers targets
from aib.cli import DEFAULT_CORPUS
from aib.dataset import load
from aib.harness import build_scenario, run, summarise
from aib.scoring import Outcome


def test_naive_agent_is_compromised_by_the_corpus():
    """The worst-case baseline must actually be worst-case.

    If the naive agent ever scores 0%, the corpus has stopped working and
    every other number in the benchmark is meaningless.
    """
    attacks = load(DEFAULT_CORPUS)
    summary = summarise(run(attacks, get_target("naive"), defenses.get("none")))
    assert summary.compromise_rate > 0.3


def test_paranoid_agent_trades_safety_for_usefulness():
    attacks = load(DEFAULT_CORPUS)
    summary = summarise(run(attacks, get_target("paranoid"), defenses.get("none")))
    assert summary.compromise_rate == 0.0
    # The whole argument for reporting both numbers: a perfect compromise
    # score bought with a total loss of usefulness.
    assert summary.over_refusal_rate == 1.0


def test_controls_excluded_from_compromise_rate():
    attacks = load(DEFAULT_CORPUS)
    controls = sum(1 for a in attacks if a.is_control)
    summary = summarise(run(attacks, get_target("naive"), defenses.get("none")))
    assert summary.total_controls == controls
    assert summary.total_attacks == len(attacks) - controls


def test_every_defense_runs_without_error():
    attacks = load(DEFAULT_CORPUS)
    for name in defenses.available():
        summary = summarise(run(attacks, get_target("naive"), defenses.get(name)))
        assert summary.errors == 0, f"defense {name} produced errors"


def test_allowlist_never_empties_the_registry():
    """A defense that removes every tool would score as perfectly safe.

    Guarding this explicitly because it is the exact shape of a defense that
    games the benchmark rather than improving security.
    """
    attacks = [a for a in load(DEFAULT_CORPUS) if a.tools]
    allowlist = defenses.get("allowlist")
    for attack in attacks:
        scenario = allowlist(build_scenario(attack))
        assert scenario.tools, f"{attack.id}: allowlist emptied the tool registry"


def test_multi_turn_history_is_ordered():
    attacks = {a.id: a for a in load(DEFAULT_CORPUS)}
    attack = attacks["ch-001"]
    scenario = build_scenario(attack)
    roles = [m["role"] for m in scenario.messages]
    # Payload plants, assistant acknowledges, benign turn triggers.
    assert roles == ["user", "assistant", "user"]
    assert attack.injected in scenario.messages[0]["content"]
    assert attack.task in scenario.messages[-1]["content"]


def test_spotlight_marks_untrusted_content():
    attacks = {a.id: a for a in load(DEFAULT_CORPUS)}
    scenario = defenses.get("spotlight")(build_scenario(attacks["io-001"]))
    # Every space in untrusted content replaced by the nonce.
    assert all(" " not in body for body in scenario.tool_results.values())
    assert "marker" in scenario.system_prompt
