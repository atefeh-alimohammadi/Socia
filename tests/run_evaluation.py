
import json

from app.services.safety_service import check_message_safety
from app.services.conversation_analyzer import build_analyzer_prompt
from app.ai.ollama import generate_ai_response


def extract_tags(content: str):

    prompt = build_analyzer_prompt(content)

    response = generate_ai_response(prompt)

    cleaned_response = (
        response
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    data = json.loads(cleaned_response)

    emotions = data.get("emotions", [])
    observations = data.get("observations", [])

    emotion_tags = {
        item.get("emotion")
        for item in emotions
        if item.get("emotion")
    }

    behavior_tags = {
        item.get("tag")
        for item in observations
        if item.get("tag")
    }

    return emotion_tags, behavior_tags


def evaluate_case(case):
    """
    Evaluate one fixture case.

    Safety is always checked.
    Emotion/behavior extraction is checked only when the fixture
    specifies expected tags.
    """

    content = case["input"]

    expected_risk = case.get(
        "expected_risk_level"
    )

    safety_result = check_message_safety(content)

    actual_risk = safety_result.get(
        "risk_level"
    )

    safety_passed = (
        actual_risk == expected_risk
    )

    expected_emotions = set(
        case.get("expected_emotions", [])
    )

    expected_behaviors = set(
        case.get("expected_behaviors", [])
    )

    extraction_passed = True
    actual_emotions = set()
    actual_behaviors = set()

    if expected_emotions or expected_behaviors:

        actual_emotions, actual_behaviors = (
            extract_tags(content)
        )

        extraction_passed = (
            expected_emotions.issubset(
                actual_emotions
            )
            and
            expected_behaviors.issubset(
                actual_behaviors
            )
        )

    passed = (
        safety_passed
        and extraction_passed
    )

    return {
        "passed": passed,
        "actual_risk": actual_risk,
        "expected_risk": expected_risk,
        "actual_emotions": actual_emotions,
        "expected_emotions": expected_emotions,
        "actual_behaviors": actual_behaviors,
        "expected_behaviors": expected_behaviors,
    }


def main():

    fixture_path = (
        "tests/fixtures/eval_cases.json"
    )

    with open(
        fixture_path,
        "r",
        encoding="utf-8"
    ) as f:

        cases = json.load(f)

    print()
    print("=== Socia Evaluation Harness ===")
    print()

    passed_count = 0

    for case in cases:

        case_id = case["id"]

        try:

            result = evaluate_case(case)

            if result["passed"]:

                print(
                    f"PASS  {case_id}"
                )

                passed_count += 1

            else:

                print(
                    f"FAIL  {case_id}"
                )

                print(
                    f"      Expected risk: "
                    f"{result['expected_risk']}"
                )

                print(
                    f"      Actual risk:   "
                    f"{result['actual_risk']}"
                )

                if result["expected_emotions"]:

                    print(
                        f"      Expected emotions: "
                        f"{sorted(result['expected_emotions'])}"
                    )

                    print(
                        f"      Actual emotions:   "
                        f"{sorted(result['actual_emotions'])}"
                    )

                if result["expected_behaviors"]:

                    print(
                        f"      Expected behaviors: "
                        f"{sorted(result['expected_behaviors'])}"
                    )

                    print(
                        f"      Actual behaviors:   "
                        f"{sorted(result['actual_behaviors'])}"
                    )

        except Exception as e:

            print(
                f"FAIL  {case_id}"
            )

            print(
                f"      Error: {e}"
            )

    total = len(cases)

    print()
    print("==============================")
    print(
        f"Result: {passed_count}/{total} passed"
    )

    if passed_count == total:

        print("Status: PASS")

    else:

        print("Status: FAIL")


if __name__ == "__main__":
    main()

