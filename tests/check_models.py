"""Measure which free-tier models actually work on the compatibility endpoint.

This is the script behind the model table in the README. Google's docs neither
publish free-tier daily quotas nor promise that
response_format={"type": "json_object"} is supported, so both were established
by asking the API instead of by reading about it.

Re-run this if the model list ever needs revisiting - Google retires models
without much warning.

Costs one API call per candidate. Run: python3 tests/check_models.py
"""

import re

import _path  # noqa: F401

from meal_planner import MODELS, build_client

CANDIDATES = [
    "gemini-3.7-flash",          # Google's documented example model
    "gemini-3.6-flash",          # current first choice
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",     # current fallback
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
]


def main():
    client = build_client()

    print("configured chain:", MODELS)

    print("\n--- models served to this key ---")
    try:
        served = sorted(m.id for m in client.models.list())
        for name in served:
            if "flash" in name or "pro" in name:
                print("  ", name)
    except Exception as exc:
        print("   models.list() failed: %s" % type(exc).__name__)

    print("\n--- json_object support, one call each ---")
    for model in CANDIDATES:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You return JSON."},
                    {"role": "user",
                     "content": 'Return JSON: {"ok": true}'},
                ],
                response_format={"type": "json_object"},
            )
            print("%-28s OK    %s"
                  % (model, resp.choices[0].message.content.strip()[:36]))

        except Exception as exc:
            message = str(exc)
            quota = re.search(r"'quotaValue': '(\d+)'", message)
            detail = "daily quota %s" % quota.group(1) if quota else ""
            print("%-28s %-22s %s"
                  % (model, type(exc).__name__, detail or message[:60]))


if __name__ == "__main__":
    main()
