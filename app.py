"""Agentic Meal Planner - Hugging Face Spaces entry point (Gradio UI).

Wraps the same orchestration loop as meal_planner.py in a web UI. The point of
the interface is to make the agentic loop *visible*: the draft, each critique,
each deterministic validation failure, and each revision are streamed to the
browser as they happen rather than hidden behind a spinner.

The API key is read from the GOOGLE_API_KEY environment variable, which on
Hugging Face Spaces should be set as a Repository Secret (Settings -> Variables
and secrets) rather than committed. Visitors never see it - only this
server-side process reads it. A visitor may optionally supply their own free
key, which is used for that request only and never stored or logged.
"""

import inspect
import os
import queue
import threading

import gradio as gr

from logo import LOGO, WIDTH as LOGO_WIDTH
from meal_planner import (
    MissingAPIKey,
    QuotaExhausted,
    build_client,
    build_grocery_list,
    build_plan,
    set_notice_hook,
    user_inputs,
)

# Sentinel pushed onto the log queue to signal that the worker is finished.
_DONE = object()


def parse_list(text):
    """Split a comma-separated field into a clean list of names."""
    if not text:
        return []

    return [item.strip() for item in text.split(",") if item.strip()]


def format_plan(plan):
    """Render the final plan as a readable Markdown table."""

    menu = plan.get("menu", [])

    if not menu:
        return "## Final plan\n\n_No plan was produced._"

    lines = [
        "## Final plan",
        "",
        "| Day | Dish | Calories (per person) | Est. cost |",
        "| --- | --- | ---: | ---: |",
    ]

    for meal in menu:
        lines.append(
            "| {day} | {dish} | {cal} | ${cost} |".format(
                day=meal.get("day", ""),
                dish=meal.get("dish", ""),
                cal=meal.get("calories", ""),
                cost=meal.get("est_cost_usd", ""),
            )
        )

    lines.append("")
    lines.append("**Total cost: ${}**".format(plan.get("total_cost", "?")))

    lines.append("")
    lines.append("### Ingredients by day")

    for meal in menu:
        lines.append(
            "- **{day}** - {ingredients}".format(
                day=meal.get("day", ""),
                ingredients=", ".join(meal.get("ingredients", [])),
            )
        )

    return "\n".join(lines)


def format_groceries(groceries):
    """Render the shopping list as a readable Markdown table."""

    items = groceries.get("shopping_list", [])

    if not items:
        return "## Grocery list\n\n_No shopping list was produced._"

    lines = [
        "## Grocery list",
        "",
        "| Item | Estimated quantity |",
        "| --- | --- |",
    ]

    for entry in items:
        lines.append(
            "| {item} | {qty} |".format(
                item=entry.get("item", ""),
                qty=entry.get("estimated_qty", ""),
            )
        )

    return "\n".join(lines)


def run_planner(
    people,
    days,
    daily_calories,
    budget_usd,
    allergens_text,
    pantry_text,
    visitor_key,
):
    """Gradio event handler.

    A full run is five to nine sequential LLM calls and can take a minute or
    more, so this is a generator: it yields the agent's log as it is produced
    instead of leaving the page blank until the end.

    build_plan() is synchronous and takes a `log` callback, so it runs on a
    worker thread while this generator drains a queue that the callback feeds.
    """

    params = {
        "people": int(people),
        "days": int(days),
        "daily_calories": int(daily_calories),
        "budget_usd": float(budget_usd),
        "allergens": parse_list(allergens_text),
        "pantry": parse_list(pantry_text),
    }

    # Blank fields are the normal case (no allergies, empty pantry), but a
    # zero-day plan is not something the model can act on.
    if params["days"] < 1:
        yield "Set at least one day.", "", "", None
        return

    try:
        client = build_client(api_key=visitor_key.strip() or None)
    except MissingAPIKey as exc:
        yield str(exc), "", "", None
        return

    messages = queue.Queue()
    result = {}

    def log(message):
        messages.put(str(message))

    def worker():
        # Runs on this thread only, so a model-downgrade notice lands in this
        # visitor's log rather than in someone else's concurrent run.
        set_notice_hook(log)

        try:
            plan = build_plan(params, log=log, client=client)
            result["plan"] = plan

            log("\n==============================")
            log("BUILDING GROCERY LIST")
            log("==============================")

            result["groceries"] = build_grocery_list(
                plan, params["pantry"], client=client
            )

            log("Done.")

        except QuotaExhausted as exc:
            result["error"] = str(exc)
            log("\nERROR: " + str(exc))

        except Exception as exc:  # surfaced in the UI rather than the console
            result["error"] = "{}: {}".format(type(exc).__name__, exc)
            log("\nERROR: " + result["error"])

        finally:
            messages.put(_DONE)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    transcript = []

    yield "Starting...", "", "", None

    while True:
        message = messages.get()

        if message is _DONE:
            break

        transcript.append(message)

        yield "\n".join(transcript), "", "", None

    thread.join()

    log_text = "\n".join(transcript)

    if "error" in result:
        yield log_text, "**Run failed.** " + result["error"], "", None
        return

    plan = result.get("plan", {})
    groceries = result.get("groceries", {})

    yield (
        log_text,
        format_plan(plan),
        format_groceries(groceries),
        {"plan": plan, "groceries": groceries},
    )


# The wordmark is LOGO_WIDTH characters wide. Rather than let it overflow and
# force horizontal scrolling, the font-size is expressed in container-query
# width units so the art always scales to exactly fit: a monospace glyph is
# ~0.6em wide, so the art spans LOGO_WIDTH * 0.6 em, meaning
# 100cqw / (LOGO_WIDTH * 0.6) per character fills the container exactly.
# Slightly under that leaves a margin. Below 700px the art would be
# unreadably small, so a plain text title is shown instead.
_LOGO_CQW = round(100.0 / (LOGO_WIDTH * 0.6) * 0.97, 3)

# Upper bound so the art cannot grow absurdly on a very wide viewport. It has
# to clear the size the art actually wants at the container's max width
# (1100px), or it would clamp the art small instead of merely capping it.
_LOGO_MAX_PX = max(16, int(_LOGO_CQW * 11) + 4)

CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
#logo-wrap {
  container-type: inline-size;
  text-align: center;
  overflow: hidden;
}
#logo-wrap pre {
  display: inline-block;
  text-align: left;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 10px;               /* fallback if cqw is unsupported */
  font-size: min(%(cqw)scqw, %(maxpx)spx);
  line-height: 1.05;
  color: #16a34a;
  white-space: pre;
  margin: 0;
  border: none;
  background: none;
  padding: 0;
}
#logo-fallback { display: none; text-align: center; }
#logo-fallback h1 { font-size: 30px; font-weight: 700; color: #16a34a; margin: 0; }
@media (max-width: 700px) {
  #logo-wrap { display: none; }
  #logo-fallback { display: block; }
}
#agent-log textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}
""" % {"cqw": _LOGO_CQW, "maxpx": _LOGO_MAX_PX}

SECRET_IS_SET = bool(os.environ.get("GOOGLE_API_KEY"))

KEY_NOTE = (
    "A key is configured on this Space, so you can just press Build. "
    "If the free-tier quota is exhausted, paste your own free key below."
    if SECRET_IS_SET
    else
    "No key is configured on this Space. Paste your own free key below to "
    "run it."
)

# Gradio moved `css` and `theme` between the Blocks constructor and launch()
# in version 6: 4.x accepts them only on Blocks, 6.x wants them only on
# launch() and warns otherwise. This has to run on both - locally on 4.44 and
# on the Space's 6.26 - so ask the installed version where they belong instead
# of guessing.
_STYLE = {"css": CSS, "theme": gr.themes.Soft()}

_LAUNCH_TAKES_STYLE = "css" in inspect.signature(gr.Blocks.launch).parameters

BLOCKS_STYLE = {} if _LAUNCH_TAKES_STYLE else _STYLE
LAUNCH_STYLE = _STYLE if _LAUNCH_TAKES_STYLE else {}

with gr.Blocks(title="Agentic Meal Planner", **BLOCKS_STYLE) as demo:

    # The art contains no HTML-special characters, so it needs no escaping;
    # it does need <pre> to keep its spacing.
    gr.HTML("<div id='logo-wrap'><pre>%s</pre></div>" % LOGO)
    gr.HTML("<div id='logo-fallback'><h1>Agentic Meal Planner</h1></div>")
    gr.Markdown(
        "An LLM drafts a weekly dinner plan, a second LLM critiques it, and "
        "**plain Python** independently validates it against hard constraints. "
        "Violations from both reviewers are fed back for revision, up to three "
        "passes. Only the Python validator decides whether the plan is really "
        "correct - the LLM's opinion can never mark a broken plan as passing.\n\n"
        "Watch the agent log below to see the loop actually correcting itself."
    )

    with gr.Row():
        with gr.Column():
            people = gr.Number(label="People", value=user_inputs["people"], precision=0)
            days = gr.Number(label="Days", value=user_inputs["days"], precision=0)
            daily_calories = gr.Number(
                label="Calories per person per day",
                value=user_inputs["daily_calories"],
                precision=0,
            )
            budget_usd = gr.Number(
                label="Total budget (USD)", value=user_inputs["budget_usd"]
            )

        with gr.Column():
            allergens_text = gr.Textbox(
                label="Allergens (comma-separated)",
                value=", ".join(user_inputs["allergens"]),
            )
            pantry_text = gr.Textbox(
                label="Pantry items you already have (comma-separated)",
                value=", ".join(user_inputs["pantry"]),
                info=(
                    "Matched by exact name: 'frozen spinach' will not accept "
                    "'fresh spinach'."
                ),
            )
            visitor_key = gr.Textbox(
                label="Your Google AI Studio API key (optional)",
                type="password",
                placeholder="Leave blank to use the Space's own key",
                info=(
                    KEY_NOTE
                    + " Free keys: https://aistudio.google.com/app/api-keys - "
                    "used for this request only, never stored."
                ),
            )

    build_btn = gr.Button("Build meal plan", variant="primary")

    gr.Markdown(
        "_A full run makes five to nine sequential model calls and usually "
        "takes 30-90 seconds._"
    )

    # No show_copy_button: Gradio 6 removed it from Textbox.
    agent_log = gr.Textbox(
        label="Agent log (live)", lines=20, elem_id="agent-log"
    )

    # gr.Markdown ignores `label`, so these carry their own headings and start
    # with placeholder text - otherwise the page shows two blank gaps before
    # the first run.
    plan_out = gr.Markdown("## Final plan\n\n_Not built yet._")
    grocery_out = gr.Markdown("## Grocery list\n\n_Not built yet._")

    with gr.Accordion("Raw JSON", open=False):
        # gr.JSON takes the object itself and renders a collapsible tree. It
        # has also been stable across Gradio majors, unlike some of its
        # neighbours.
        raw_out = gr.JSON()

    build_btn.click(
        run_planner,
        inputs=[
            people,
            days,
            daily_calories,
            budget_usd,
            allergens_text,
            pantry_text,
            visitor_key,
        ],
        outputs=[agent_log, plan_out, grocery_out, raw_out],
    )


if __name__ == "__main__":
    demo.launch(**LAUNCH_STYLE)
