"""Drive a real run through the web UI and screenshot the finished page.

Produces screenshots/app.png. This is an end-to-end test as much as a capture:
it clicks the button, waits for the grocery table to render, and fails loudly
if the run never completes or the browser logs an error.

Requires the app to be running, and playwright with chromium installed:

    python3 app.py &
    python3 tests/capture_screenshot.py [port] [output.png]
"""

import os
import sys

import _path  # noqa: F401
from playwright.sync_api import sync_playwright

PORT = sys.argv[1] if len(sys.argv) > 1 else "7860"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    _path.ROOT, "screenshots", "app.png"
)
URL = "http://127.0.0.1:%s/" % PORT


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        errors = []
        page.on("console",
                lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(URL, wait_until="networkidle", timeout=60000)
        page.wait_for_selector("button:has-text('Build meal plan')",
                               timeout=30000)

        page.click("button:has-text('Build meal plan')")
        print("clicked build; waiting for the run to finish...")

        completed = True
        try:
            # The run is done once the grocery table has rendered.
            page.wait_for_selector("text=Estimated quantity", timeout=300000)
        except Exception:
            completed = False
            print("run did not finish in time")

        page.wait_for_timeout(3000)

        # Show the end of the streamed log rather than its first lines.
        page.evaluate(
            "document.querySelectorAll('#agent-log textarea')"
            ".forEach(t => t.scrollTop = t.scrollHeight)"
        )
        page.wait_for_timeout(500)

        page.screenshot(path=OUT, full_page=True)
        print("saved:", OUT)

        body = page.inner_text("body")
        has_plan = "Final plan" in body and "Grocery list" in body

        print("console errors:", errors[:5] if errors else "none")

        browser.close()

    ok = completed and has_plan and not errors
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
