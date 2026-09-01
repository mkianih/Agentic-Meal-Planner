"""Force the correction loop to run, by seeding a deliberately broken draft.

The model writes valid plans on the first attempt almost every time, so
revise_plan() never executes in a normal run and would otherwise ship
untested. Monkeypatching draft_plan is the only reliable way to exercise it.

Costs several API calls. Run: python3 tests/test_correction_loop.py
"""

import _path  # noqa: F401

import meal_planner
from meal_planner import user_inputs, validate_plan

# Breaks five rules at once: too few days, over budget, calories far too high,
# pantry names wrong ("basmati rice" / "fresh spinach"), and two allergens.
BROKEN = {
    "menu": [
        {"day": "Monday", "dish": "Peanut Chicken",
         "ingredients": ["chicken", "peanut oil", "basmati rice"],
         "calories": 3200, "est_cost_usd": 42},
        {"day": "Tuesday", "dish": "Spinach Pasta",
         "ingredients": ["pasta", "fresh spinach", "cream"],
         "calories": 3200, "est_cost_usd": 42},
        {"day": "Wednesday", "dish": "Steak",
         "ingredients": ["steak", "butter"],
         "calories": 3200, "est_cost_usd": 42},
        {"day": "Thursday", "dish": "Shrimp Scampi",
         "ingredients": ["shrimp", "shellfish stock", "pasta"],
         "calories": 3200, "est_cost_usd": 42},
        {"day": "Friday", "dish": "Lamb Chops",
         "ingredients": ["lamb", "potatoes"],
         "calories": 3200, "est_cost_usd": 42},
    ],
    "total_cost": 210,
}


def main():
    print("--- violations in the seeded draft ---")
    for problem in validate_plan(BROKEN, user_inputs):
        print("  -", problem)

    meal_planner.draft_plan = lambda params, client=None: dict(BROKEN)

    lines = []

    def log(message):
        text = str(message)
        lines.append(text)
        if len(text) < 200 and any(
            k in text for k in
            ("PASS", "VALIDATION", "violations", "WARNING", "REVISED",
             "passed")
        ):
            print(text)

    print("\n--- running build_plan ---")
    plan = meal_planner.build_plan(user_inputs, log=log)

    revised = any("REVISED PLAN" in x for x in lines)
    problems = validate_plan(plan, user_inputs)

    print("\n--- result ---")
    print("revise_plan was invoked:", revised)
    print("remaining problems:", len(problems))
    for problem in problems:
        print("  -", problem)

    ok = revised and not problems
    print("\n%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
