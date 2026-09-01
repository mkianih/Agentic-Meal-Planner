"""Verify the model fallback chain and the thread-local notice hook.

Rather than waiting a day for a real quota reset, this puts a model that is
reliably unavailable on the free tier at the head of the chain and checks that
the call still succeeds on the next one.

Costs a few API calls. Run: python3 tests/test_fallback.py
"""

import threading

import _path  # noqa: F401

import meal_planner

# Not served on the free tier: returns 503 or 429 on every attempt, which makes
# it a dependable stand-in for an exhausted primary model.
UNAVAILABLE = "gemini-3.7-flash"


def main():
    original = list(meal_planner.MODELS)
    print("configured chain:", original)

    results = []

    # --- falls through to the next model -----------------------------------
    meal_planner.MODELS = [UNAVAILABLE, original[-1]]

    notices = []
    meal_planner.set_notice_hook(notices.append)

    raw = meal_planner.get_completion(
        'Return JSON: {"dish": "soup"}', "You return JSON.", json_mode=True
    )
    parsed = meal_planner.parse_json_response(raw)

    ok = parsed.get("dish") == "soup"
    results.append(ok)
    print("\n%-46s %s" % ("falls back and still returns valid JSON",
                          "PASS" if ok else "FAIL"))

    ok = len(notices) == 1 and "Falling back" in notices[0]
    results.append(ok)
    print("%-46s %s" % ("downgrade is announced, not silent",
                        "PASS" if ok else "FAIL"))
    for n in notices:
        print("      ", n.strip())

    # --- the hook must not leak into other threads -------------------------
    seen = []
    thread = threading.Thread(
        target=lambda: seen.append(
            getattr(meal_planner._local, "hook", None))
    )
    thread.start()
    thread.join()

    ok = seen[0] is None
    results.append(ok)
    print("%-46s %s" % ("notice hook is thread-local",
                        "PASS" if ok else "FAIL"))

    # --- exhausting the chain raises the friendly error --------------------
    meal_planner.MODELS = [UNAVAILABLE]
    try:
        meal_planner.get_completion("hi", "hi")
        ok = False
        print("%-46s FAIL (call unexpectedly succeeded)"
              % "all models exhausted -> QuotaExhausted")
    except meal_planner.QuotaExhausted:
        ok = True
        print("%-46s PASS" % "all models exhausted -> QuotaExhausted")
    except Exception as exc:
        ok = False
        print("%-46s FAIL (leaked %s)"
              % ("all models exhausted -> QuotaExhausted",
                 type(exc).__name__))
    results.append(ok)

    meal_planner.MODELS = original

    print("\n%d/%d passed" % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
