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

import json
import os
import queue
import threading

import gradio as gr

from meal_planner import (
    MissingAPIKey,
    QuotaExhausted,
    build_client,
    build_grocery_list,
    build_plan,
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
        return "_No plan was produced._"

    lines = [
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
        return "_No shopping list was produced._"

    lines = [
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
        yield "Set at least one day.", "", "", ""
        return

    try:
        client = build_client(api_key=visitor_key.strip() or None)
    except MissingAPIKey as exc:
        yield str(exc), "", "", ""
        return

    messages = queue.Queue()
    result = {}

    def log(message):
        messages.put(str(message))

    def worker():
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

    yield "Starting...", "", "", ""

    while True:
        message = messages.get()

        if message is _DONE:
            break

        transcript.append(message)

        yield "\n".join(transcript), "", "", ""

    thread.join()

    log_text = "\n".join(transcript)

    if "error" in result:
        yield log_text, "**Run failed.** " + result["error"], "", ""
        return

    plan = result.get("plan", {})
    groceries = result.get("groceries", {})

    yield (
        log_text,
        format_plan(plan),
        format_groceries(groceries),
        json.dumps({"plan": plan, "groceries": groceries}, indent=2),
    )


CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
#agent-log textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}
"""

SECRET_IS_SET = bool(os.environ.get("GOOGLE_API_KEY"))

KEY_NOTE = (
    "A key is configured on this Space, so you can just press Build. "
    "If the free-tier quota is exhausted, paste your own free key below."
    if SECRET_IS_SET
    else
    "No key is configured on this Space. Paste your own free key below to "
    "run it."
)

with gr.Blocks(title="Agentic Meal Planner", css=CSS, theme=gr.themes.Soft()) as demo:

    gr.Markdown("# Agentic Meal Planner")
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

    agent_log = gr.Textbox(
        label="Agent log (live)", lines=20, elem_id="agent-log", show_copy_button=True
    )

    plan_out = gr.Markdown(label="Final plan")
    grocery_out = gr.Markdown(label="Grocery list")

    with gr.Accordion("Raw JSON", open=False):
        raw_out = gr.Code(language="json")

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
    demo.launch()
