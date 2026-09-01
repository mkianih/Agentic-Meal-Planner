"""Agentic Meal Planner - manual agentic orchestration with the OpenAI SDK.

Builds a weekly dinner plan that satisfies five hard constraints (day count,
budget, calories, allergens, pantry usage) by chaining LLM calls and then
correcting them in a loop:

    draft -> critique (LLM) + validate (pure Python) -> revise -> repeat

The novel part is the split between the two reviewers. `critique_plan()` is an
LLM asked for a subjective opinion; `validate_plan()` is deterministic Python
that cannot hallucinate. Only the Python validator is trusted to decide whether
the plan is actually correct - the LLM's critique only ever adds work.

No agent framework is used. The control flow in `build_plan()` is plain Python.

Provider note: this runs on Google's free Gemini tier through its
OpenAI-compatible endpoint, so every call below is ordinary OpenAI SDK code -
`client.chat.completions.create(...)` with `response_format={"type":
"json_object"}`. Only the base_url, API key, and model name differ.
"""

import json
import os
import re
import threading

from dotenv import load_dotenv
from openai import (
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

# Loads a local .env for development. On Hugging Face Spaces there is no .env
# and the key arrives as a repository secret instead, so a failure here must
# never stop the module from importing.
try:
    load_dotenv()
except Exception:
    pass


# ============================================================
# PROVIDER CONFIGURATION
# ============================================================

# Only "google" is active, and every model it can reach is on the free tier.
# There is deliberately no model picker anywhere in this project, so no code
# path and no user action can select a billable model.
PROVIDER = "google"

PROVIDERS = {
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GOOGLE_API_KEY",
        # Ordered best-first; a call falls through to the next model when one
        # is rate-limited or unavailable. All three were verified to accept
        # response_format={"type": "json_object"} on this endpoint.
        #
        # Chosen by measurement rather than from the docs:
        #   gemini-3.7-flash    Google's own example model. Returned 503 and
        #                       then 429 on every single attempt - it does not
        #                       appear to be served on the free tier. Excluded.
        #   gemini-3.6-flash    Best output by a wide margin: varied dishes,
        #                       plausible per-day calories and costs. Capped at
        #                       20 requests/day, and one planner run costs 3-8,
        #                       so a shared Space exhausts it in ~2 runs.
        #   gemini-3.5-flash-lite  Larger daily quota, weaker planning: left to
        #                       itself it served lentils, rice and spinach all
        #                       seven nights. Good enough as a fallback, and a
        #                       live demo beats a dead one.
        "models": [
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
        ],
    },
    # Paid, intentionally left inactive. This was the original target of the
    # exercise, and it is kept here only to show that switching costs one
    # constant - the SDK calls further down do not change at all.
    # "openai": {
    #     "base_url": None,
    #     "api_key_env": "OPENAI_API_KEY",
    #     "model": "gpt-4o",
    # },
}

BASE_URL = PROVIDERS[PROVIDER]["base_url"]
API_KEY_ENV = PROVIDERS[PROVIDER]["api_key_env"]
MODELS = PROVIDERS[PROVIDER]["models"]

# The preferred model. Kept as a single name because that is what the rest of
# the program talks about; MODELS is only consulted when one is unavailable.
MODEL = MODELS[0]


# Progress notices (such as a model downgrade) should reach whichever front end
# is running. The web UI serves each run on its own worker thread, so the hook
# is thread-local - two visitors cannot end up writing into each other's logs.
_local = threading.local()


def set_notice_hook(hook):
    """Route notices to a callable for the current thread. Defaults to print."""
    _local.hook = hook


def _notice(message):
    hook = getattr(_local, "hook", None)

    if hook is None:
        print(message)
    else:
        hook(message)


class MissingAPIKey(RuntimeError):
    """Raised when no API key is configured, with instructions for both
    local runs and Hugging Face Spaces."""


class QuotaExhausted(RuntimeError):
    """Raised when the provider rate-limits us. Separated from other errors so
    the UI can suggest supplying a personal key instead of showing a
    traceback."""


def build_client(api_key=None):
    """Build an OpenAI SDK client pointed at the configured provider.

    `api_key` lets the web UI pass a visitor-supplied key for a single
    request without touching the process-wide default client.
    """
    key = api_key or os.environ.get(API_KEY_ENV)

    if not key:
        raise MissingAPIKey(
            "{env} is not set. Locally, put it in a .env file "
            "({env}=your-key-here). On Hugging Face Spaces, set it under "
            "Settings -> Variables and secrets -> New secret. "
            "A free key is available at "
            "https://aistudio.google.com/app/api-keys".format(env=API_KEY_ENV)
        )

    kwargs = {"api_key": key}

    # A None base_url would override the SDK's own default, so only pass it
    # when the provider actually needs a custom endpoint.
    if BASE_URL:
        kwargs["base_url"] = BASE_URL

    return OpenAI(**kwargs)


# Built on first use rather than at import time, so that importing this module
# (for tests, or for the UI's static layout) does not require a key.
_default_client = None


def get_default_client():
    global _default_client

    if _default_client is None:
        _default_client = build_client()

    return _default_client


# ============================================================
# GENERAL LLM COMPLETION FUNCTION
# ============================================================

def get_completion(
    prompt,
    system_prompt="You are a helpful assistant.",
    json_mode=False,
    client=None
):
    """Single-turn chat completion.

    The first three parameters are the interface every step below uses. The
    optional `client` exists only so the web UI can inject a per-request
    client built from a visitor's own API key.
    """

    active_client = client or get_default_client()

    # Built conditionally on purpose. The SDK's default for response_format is
    # `Omit`, not None - passing None explicitly serializes
    # "response_format": null into the request body and the API rejects it.
    request = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    }

    if json_mode:
        request["response_format"] = {"type": "json_object"}

    try:
        response = _create_completion(active_client, request, json_mode)

    # Both failure modes are raised only after every model in the chain has
    # been tried, and both mean the same thing to whoever is waiting: nothing
    # is available right now. Converting them here keeps a traceback off the
    # screen.
    except (RateLimitError, InternalServerError) as exc:
        raise QuotaExhausted(
            "Every available model is currently rate-limited or overloaded. "
            "The free tier allows only a small number of requests per day per "
            "model, and one meal plan uses several. Wait for the quota to "
            "reset, or supply your own API key."
        ) from exc

    return response.choices[0].message.content


def _create_completion(client, request, json_mode):
    """Send the request, trying each model in turn.

    The free tier meters requests per day *per model*, so when the preferred
    model is exhausted the next one still has its own budget. Falling through
    keeps a shared demo alive instead of failing outright; the downgrade is
    announced so the output is never silently worse than it looks.
    """

    for index, model in enumerate(MODELS):

        attempt = dict(request)
        attempt["model"] = model

        try:
            return _create_once(client, attempt, json_mode)

        except (RateLimitError, InternalServerError) as exc:

            is_last = index + 1 >= len(MODELS)

            if is_last:
                raise

            _notice(
                "\nNote: {model} is unavailable ({reason}). "
                "Falling back to {next_model}, whose meal plans are "
                "noticeably less varied.".format(
                    model=model,
                    reason=type(exc).__name__,
                    next_model=MODELS[index + 1],
                )
            )

    # Unreachable: the loop either returns or raises on the final model.
    raise RuntimeError("No models are configured.")


def _create_once(client, request, json_mode):
    """One model, with a graceful degradation if it rejects JSON mode.

    Google's OpenAI-compatibility layer does not document
    `{"type": "json_object"}` (it demonstrates Pydantic parsing instead).
    Every model configured here was verified to accept it, but if a future one
    does not, the prompts already demand JSON-only output and
    `parse_json_response()` already strips stray Markdown fences - so retrying
    without the parameter still works, just without the server-side guarantee.
    """

    try:
        return client.chat.completions.create(**request)

    except BadRequestError as exc:

        if not json_mode or "response_format" not in str(exc):
            raise

        _notice(
            "\nNote: {} rejected response_format=json_object. Retrying "
            "without it and relying on prompt-enforced JSON.".format(
                request.get("model")
            )
        )

        fallback = dict(request)
        fallback.pop("response_format", None)

        return client.chat.completions.create(**fallback)


# ============================================================
# USER INPUTS
# ============================================================

user_inputs = {
    "people": 2,
    "days": 7,
    "daily_calories": 2000,
    "allergens": [
        "peanuts",
        "shellfish"
    ],
    "budget_usd": 110,
    "pantry": [
        "rice",
        "lentils",
        "frozen spinach"
    ]
}


# ============================================================
# MEAL PLAN JSON SCHEMA
# ============================================================

schema_string = """
{
  "menu": [
    {
      "day": "",
      "dish": "",
      "ingredients": [],
      "calories": 0,
      "est_cost_usd": 0
    }
  ],
  "total_cost": 0
}
"""


# ============================================================
# HELPER: SAFELY PARSE JSON
# ============================================================

def parse_json_response(raw):
    """
    Safely parses JSON returned by the LLM.

    Handles:
    - empty responses
    - surrounding whitespace
    - accidental ```json code fences
    """

    if raw is None:
        raise ValueError("The model returned None instead of JSON.")

    raw = raw.strip()

    if not raw:
        raise ValueError("The model returned an empty response.")

    # Remove Markdown fences if the model accidentally adds them
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]

    if raw.endswith("```"):
        raw = raw[:-3]

    raw = raw.strip()

    try:
        return json.loads(raw)

    except json.JSONDecodeError as e:

        print("\nINVALID JSON RESPONSE FROM MODEL:")
        print(raw)

        raise ValueError(
            "Could not parse model response as JSON: {}".format(e)
        )


# ============================================================
# STEP 1
# DRAFT PLAN
# ============================================================

def draft_plan(params, client=None):

    system_prompt = """
You are a registered dietician who writes weekly dinner plans.

You must follow all dietary, budget, pantry, and calorie constraints.
Always return valid JSON when requested.
"""

    user_prompt = """
<instructions>

Your task is to create a {days}-day dinner plan
for a family of {people} people.

Requirements:

1. Assign around {daily_calories} calories PER PERSON each day.

   The "calories" field for each day means calories per person,
   not the combined total for the whole family.

2. Keep the estimated total cost less than or equal to
   ${budget_usd}.

3. The family has the following food allergies:

   {allergens}

   Do NOT include any of these allergens.

4. The following ingredients are already available in the pantry:

   {pantry}

IMPORTANT:

Use every pantry ingredient at least once during the week.

Write pantry ingredients using EXACTLY the names listed above.

Write "rice", not "basmati rice" or "cooked rice".
Write "frozen spinach", not "fresh spinach" or "spinach".

You may list other ingredients however you like. This exact-name rule
applies only to the pantry ingredients.

Use as many pantry ingredients as reasonably possible while still
meeting all other requirements.

</instructions>

<schema>

Return ONLY valid JSON.

Do not include:
- Markdown
- code fences
- explanations
- comments

The JSON must match this schema:

{schema}

</schema>
""".format(
        days=params["days"],
        people=params["people"],
        daily_calories=params["daily_calories"],
        budget_usd=params["budget_usd"],
        allergens=", ".join(params["allergens"]),
        pantry=", ".join(params["pantry"]),
        schema=schema_string,
    )

    raw_json = get_completion(
        user_prompt,
        system_prompt,
        json_mode=True,
        client=client
    )

    return parse_json_response(raw_json)


# ============================================================
# STEP 2
# CRITIQUE PLAN
# ============================================================

def critique_plan(plan, params, client=None):

    system_prompt = """
You are a strict dietary quality-assurance inspector.

Carefully inspect every constraint.

Do not assume similar ingredients are equivalent.

For example:
"fresh spinach" is NOT the same pantry ingredient as
"frozen spinach".

Always return valid JSON.
"""

    user_prompt = """
<instructions>

Here is the proposed meal plan:

{plan}

Check the following rules carefully.

RULE 1 - BUDGET

The total_cost must be less than or equal to:

${budget_usd}


RULE 2 - CALORIES

Each day's calories must be within +/-15% of:

{daily_calories} calories per person.


RULE 3 - ALLERGENS

NONE of these allergens may appear:

{allergens}


RULE 4 - PANTRY INGREDIENTS

Every pantry ingredient below must appear at least once,
written with its exact name:

{pantry}

Ingredient matching must be strict.

For example:

"frozen spinach"

does NOT count as:

"fresh spinach"


RULE 5 - PLAN LENGTH

There must be exactly:

{days} days.


Place every actual rule violation inside "fixes".

Then provide optional improvements in "suggestions",
such as:

- better variety
- easier preparation
- seasonal vegetables
- better pantry usage
- nutritional balance

If there are no mandatory problems, "fixes" must be [].

If there are no worthwhile optional improvements,
"suggestions" may also be [].

</instructions>

<schema>

Return ONLY valid JSON exactly in this format:

{{
  "fixes": [],
  "suggestions": []
}}

Do not output Markdown.
Do not output comments.
Do not add text outside the JSON.

</schema>
""".format(
        plan=json.dumps(plan, indent=2),
        budget_usd=params["budget_usd"],
        daily_calories=params["daily_calories"],
        allergens=", ".join(params["allergens"]),
        pantry=", ".join(params["pantry"]),
        days=params["days"],
    )

    raw = get_completion(
        user_prompt,
        system_prompt,
        json_mode=True,
        client=client
    )

    return parse_json_response(raw)


# ============================================================
# STEP 3
# REVISE PLAN
# ============================================================

def revise_plan(
    plan,
    fixes,
    suggestions,
    params,
    client=None
):

    system_prompt = """
You are a senior meal planner applying corrections.

Your job is to revise meal plans according to QA feedback.

Mandatory fixes must always be applied.

Optional suggestions should only be applied when they do not
violate dietary, calorie, pantry, or budget constraints.

Always return valid JSON.
"""

    user_prompt = """
<instructions>

Here is the current meal plan:

{plan}


MANDATORY FIXES:

{fixes}


OPTIONAL SUGGESTIONS:

{suggestions}


Apply every mandatory fix.

Use optional suggestions only if they do not break any rules.

The final plan must satisfy all of these requirements:

- Number of people:
  {people}

- Number of days:
  {days}

- Daily calorie target, per person:
  {daily_calories}

- Maximum total budget:
  ${budget_usd}

- Allergens that must never appear:
  {allergens}

- Pantry ingredients that must all appear:
  {pantry}

Use the exact pantry ingredient names.

If the pantry contains "frozen spinach", write "frozen spinach"
and not "fresh spinach". If it contains "rice", write "rice"
and not "basmati rice".

Return the updated plan only.

</instructions>

<schema>

Return ONLY valid JSON matching exactly this schema:

{schema}

Do not include Markdown.
Do not include explanations.
Do not include comments.

</schema>
""".format(
        plan=json.dumps(plan, indent=2),
        fixes=json.dumps(fixes, indent=2),
        suggestions=json.dumps(suggestions, indent=2),
        people=params["people"],
        days=params["days"],
        daily_calories=params["daily_calories"],
        budget_usd=params["budget_usd"],
        allergens=", ".join(params["allergens"]),
        pantry=", ".join(params["pantry"]),
        schema=schema_string,
    )

    raw = get_completion(
        user_prompt,
        system_prompt,
        json_mode=True,
        client=client
    )

    return parse_json_response(raw)


# ============================================================
# DETERMINISTIC VALIDATION
# ============================================================

def _allergen_variants(allergen):
    """Singular and plural forms of an allergen name.

    The pantry list says "peanuts", but a plan is far more likely to contain
    "peanut butter" or "peanut oil". A plain substring test for "peanuts"
    misses all of those, and a missed allergen is the one failure in this
    program with real-world consequences - so it is worth matching both forms.
    """

    name = allergen.strip().lower()

    variants = set([name])

    if name.endswith("es"):
        variants.add(name[:-2])

    if name.endswith("s"):
        variants.add(name[:-1])
    else:
        variants.add(name + "s")

    return variants


def _contains_allergen(ingredient, allergen):
    """Whole-word allergen match.

    Word boundaries matter in both directions: "peanut" must match
    "peanut butter", but "egg" must not match "eggplant".
    """

    for variant in _allergen_variants(allergen):

        pattern = r"\b" + re.escape(variant) + r"\b"

        if re.search(pattern, ingredient):
            return True

    return False


def validate_plan(plan, params):
    """
    Uses normal Python logic for constraints that can be checked
    deterministically. No LLM is involved, so this cannot hallucinate
    a passing grade.

    Returns a list of detected problems.
    """

    problems = []

    # --------------------------------------------------------
    # Check plan length
    # --------------------------------------------------------

    menu = plan.get("menu", [])

    if len(menu) != params["days"]:
        problems.append(
            "Plan has {actual} days instead of {expected}.".format(
                actual=len(menu),
                expected=params["days"]
            )
        )

    # --------------------------------------------------------
    # Check budget
    # --------------------------------------------------------

    total_cost = plan.get("total_cost", 0)

    if total_cost > params["budget_usd"]:
        problems.append(
            "Total cost ${actual} exceeds the budget of ${budget}.".format(
                actual=total_cost,
                budget=params["budget_usd"]
            )
        )

    # --------------------------------------------------------
    # Check calories
    # --------------------------------------------------------

    target = params["daily_calories"]

    lower_bound = target * 0.85
    upper_bound = target * 1.15

    for meal in menu:

        calories = meal.get("calories", 0)
        day = meal.get("day", "Unknown day")

        if calories < lower_bound or calories > upper_bound:

            problems.append(
                "{day} has {calories} calories. "
                "Allowed range is {low:.0f}-{high:.0f}.".format(
                    day=day,
                    calories=calories,
                    low=lower_bound,
                    high=upper_bound
                )
            )

    # --------------------------------------------------------
    # Gather all ingredients
    # --------------------------------------------------------

    all_ingredients = []

    for meal in menu:

        ingredients = meal.get("ingredients", [])

        for ingredient in ingredients:
            all_ingredients.append(
                ingredient.strip().lower()
            )

    # A staple like "rice" shows up on most days, so checking every occurrence
    # would report the same violation once per day. Those duplicates would then
    # be fed straight into the revision prompt as separate fixes. Check each
    # distinct ingredient once instead, preserving order for readable output.
    unique_ingredients = []

    for ingredient in all_ingredients:
        if ingredient not in unique_ingredients:
            unique_ingredients.append(ingredient)

    # --------------------------------------------------------
    # Check allergens
    # --------------------------------------------------------

    for allergen in params["allergens"]:

        for ingredient in unique_ingredients:

            if _contains_allergen(ingredient, allergen):

                problems.append(
                    "Allergen '{allergen}' appears in "
                    "ingredient '{ingredient}'.".format(
                        allergen=allergen,
                        ingredient=ingredient
                    )
                )

    # --------------------------------------------------------
    # Check pantry usage
    # --------------------------------------------------------

    # Exact normalized match, deliberately. "rice" does not accept
    # "basmati rice", and "frozen spinach" does not accept "fresh spinach".
    # This strictness is what actually gives the revision loop something to
    # correct on the first pass.
    for pantry_item in params["pantry"]:

        pantry_lower = pantry_item.strip().lower()

        if pantry_lower not in all_ingredients:

            problems.append(
                "Pantry ingredient '{item}' is not used in the plan "
                "under its exact name.".format(item=pantry_item)
            )

    return problems


# ============================================================
# STEP 4
# BUILD PLAN / ORCHESTRATION
# ============================================================

MAX_PASSES = 3


def build_plan(params, log=print, client=None):
    """Draft, then critique/validate/revise until the plan is clean.

    `log` defaults to print for command-line use. The web UI passes a callable
    that forwards the same messages to the browser, so both front ends see
    identical reasoning without this function knowing which one is running.
    """

    # --------------------------------------------------------
    # Initial draft
    # --------------------------------------------------------

    plan = draft_plan(params, client=client)

    log("\n==============================")
    log("INITIAL DRAFT")
    log("==============================")

    log(json.dumps(plan, indent=2))

    # --------------------------------------------------------
    # Evaluation / revision loop
    # --------------------------------------------------------

    for pass_number in range(MAX_PASSES):

        log("\n========== PASS {} ==========".format(pass_number + 1))

        # Subjective reviewer: an LLM asked for its opinion.
        critique = critique_plan(plan, params, client=client)

        log("\nLLM CRITIQUE")

        log(json.dumps(critique, indent=2))

        # Objective reviewer: plain Python that cannot be talked out of a
        # violation. This is the one that decides correctness.
        code_problems = validate_plan(plan, params)

        log("\nPYTHON VALIDATION")

        if code_problems:

            for problem in code_problems:
                log("- " + problem)

        else:
            log("No deterministic violations found.")

        # ----------------------------------------------------
        # Combine LLM fixes + Python fixes
        # ----------------------------------------------------

        combined_fixes = list(critique.get("fixes", []))

        combined_fixes.extend(code_problems)

        suggestions = critique.get("suggestions", [])

        # ----------------------------------------------------
        # Stop if everything is acceptable
        # ----------------------------------------------------

        if not combined_fixes:

            log("\nNo mandatory violations remain.")

            break

        # ----------------------------------------------------
        # Revise
        # ----------------------------------------------------

        plan = revise_plan(
            plan,
            combined_fixes,
            suggestions,
            params,
            client=client
        )

        log("\nREVISED PLAN")

        log(json.dumps(plan, indent=2))

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    final_problems = validate_plan(plan, params)

    if final_problems:

        log("\nWARNING: FINAL PLAN STILL HAS ISSUES:")

        for problem in final_problems:
            log("- " + problem)

    else:

        log("\nFinal plan passed deterministic validation.")

    return plan


# ============================================================
# BUILD GROCERY LIST
# ============================================================

def build_grocery_list(
    plan,
    pantry,
    client=None
):

    system_prompt = """
You are a helpful kitchen assistant.

Create practical grocery lists from meal plans.

Always return valid JSON only.

Never return Markdown or code fences.
"""

    user_prompt = """
<instructions>

Create one combined grocery shopping list for the meal plan below.

The following ingredients are already available in the pantry:

{pantry}

Do NOT include those pantry ingredients in the shopping list.

Combine repeated ingredients into one shopping-list item whenever
possible.

Estimate useful quantities for the entire week.

For example:

"chicken breast": "1.2 kg"

rather than listing chicken separately for several days.

Meal plan:

{plan}

</instructions>

<schema>

Return ONLY valid JSON in exactly this format:

{{
  "shopping_list": [
    {{
      "item": "",
      "estimated_qty": ""
    }}
  ]
}}

Do not include Markdown.
Do not include comments.
Do not add explanatory text.

</schema>
""".format(
        pantry=", ".join(pantry),
        plan=json.dumps(plan, indent=2),
    )

    raw = get_completion(
        user_prompt,
        system_prompt,
        json_mode=True,
        client=client
    )

    return parse_json_response(raw)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Build meal plan
    # --------------------------------------------------------

    final_plan = build_plan(user_inputs)

    print("\n\n==============================")
    print("FINAL PLAN")
    print("==============================")

    print(json.dumps(final_plan, indent=2))

    # --------------------------------------------------------
    # Build grocery list
    # --------------------------------------------------------

    groceries = build_grocery_list(
        final_plan,
        user_inputs["pantry"]
    )

    print("\n\n==============================")
    print("GROCERY LIST")
    print("==============================")

    print(json.dumps(groceries, indent=2))
