"""Adversarial tests for the deterministic validator.

The most important tests in the project, and the only ones that need no API
key and no quota: validate_plan() is the component that decides whether a plan
is actually correct, so it must catch violations the model may introduce.

Run:  python3 tests/test_validator.py
"""

import _path  # noqa: F401  (side effect: puts the project root on sys.path)

from meal_planner import user_inputs, validate_plan

GOOD = ["rice", "lentils", "frozen spinach", "carrot"]


def day(name, calories, ingredients, cost=10):
    return {
        "day": name,
        "dish": "dish",
        "ingredients": ingredients,
        "calories": calories,
        "est_cost_usd": cost,
    }


def plan(days=7, total=100, calories=2000, ingredients=None):
    ingredients = ingredients or GOOD
    return {
        "menu": [day("D%d" % i, calories, ingredients) for i in range(days)],
        "total_cost": total,
    }


def check(name, built, params, want_problem):
    """want_problem=True means at least one problem is expected."""
    problems = validate_plan(built, params)
    found = len(problems) > 0
    ok = found == want_problem
    print("%-42s %-4s %d problem(s)"
          % (name, "PASS" if ok else "FAIL", len(problems)))
    for p in problems[:2]:
        print("      -", p)
    return ok


def main():
    results = []

    results.append(check(
        "valid plan is accepted", plan(), user_inputs, False))
    results.append(check(
        "wrong day count", plan(days=6), user_inputs, True))
    results.append(check(
        "over budget", plan(total=130), user_inputs, True))
    results.append(check(
        "calories above range", plan(calories=3000), user_inputs, True))
    results.append(check(
        "calories below range", plan(calories=1200), user_inputs, True))

    # Pantry names are matched exactly, on purpose.
    results.append(check(
        "'fresh spinach' rejected for 'frozen spinach'",
        plan(ingredients=["rice", "lentils", "fresh spinach"]),
        user_inputs, True))
    results.append(check(
        "'basmati rice' rejected for 'rice'",
        plan(ingredients=["basmati rice", "lentils", "frozen spinach"]),
        user_inputs, True))

    # Allergens: the pantry says "peanuts", the plan says "peanut oil".
    # A naive substring test for the plural misses this entirely.
    results.append(check(
        "singular 'peanut oil' vs allergen 'peanuts'",
        plan(ingredients=GOOD + ["peanut oil"]), user_inputs, True))
    results.append(check(
        "'shellfish stock' caught",
        plan(ingredients=GOOD + ["shellfish stock"]), user_inputs, True))

    # ...but word boundaries must stop the obvious false positive.
    egg = dict(user_inputs)
    egg["allergens"] = ["egg"]
    results.append(check(
        "'eggplant' does NOT trip allergen 'egg'",
        plan(ingredients=GOOD + ["eggplant"]), egg, False))
    results.append(check(
        "'egg yolk' does trip allergen 'egg'",
        plan(ingredients=GOOD + ["egg yolk"]), egg, True))

    # One offending ingredient repeated across days is one finding, not seven.
    repeated = validate_plan(
        plan(ingredients=GOOD + ["peanut oil"]), user_inputs)
    allergen_findings = [p for p in repeated if "Allergen" in p]
    ok = len(allergen_findings) == 1
    results.append(ok)
    print("%-42s %-4s %d allergen finding(s), want 1"
          % ("daily repeats are deduplicated",
             "PASS" if ok else "FAIL", len(allergen_findings)))

    print("\n%d/%d passed" % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
