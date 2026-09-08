import requests


# ============================================================
# AI SERVICE TEST CLIENT
# ============================================================

BASE_URL = "http://127.0.0.1:8000"


def test_health():

    print("\n====================================")
    print("TEST 1: HEALTH CHECK")
    print("====================================")

    response = requests.get(
        f"{BASE_URL}/health"
    )

    print("Status:", response.status_code)
    print("Response:")
    print(response.json())


def test_carbon_prediction():

    print("\n====================================")
    print("TEST 2: CARBON PREDICTION")
    print("====================================")

    data = {

        "year": 2026,

        "month": 9,

        "time_index": 24321,

        "lag_1": 100.5,

        "lag_2": 98.2,

        "lag_3": 95.7,

        "rolling_mean_3": 98.13

    }

    response = requests.post(

        f"{BASE_URL}/predict",

        json=data

    )

    print("Status:", response.status_code)

    print("Response:")
    print(response.json())


def test_subject_question():

    print("\n====================================")
    print("TEST 3: SUBJECT-SPECIFIC AI")
    print("====================================")

    data = {

        "selected_subject":
            "Carbon Emissions",

        "question":
            "What are Scope 1 emissions?"

    }

    response = requests.post(

        f"{BASE_URL}/ask",

        json=data

    )

    print("Status:", response.status_code)

    print("Response:")
    print(response.json())


def test_off_topic_question():

    print("\n====================================")
    print("TEST 4: OFF-TOPIC QUESTION")
    print("====================================")

    data = {

        "selected_subject":
            "Carbon Emissions",

        "question":
            "How do I create a Python class?"

    }

    response = requests.post(

        f"{BASE_URL}/ask",

        json=data

    )

    print("Status:", response.status_code)

    print("Response:")
    print(response.json())


def test_supply_chain_question():

    print("\n====================================")
    print("TEST 5: SUPPLY CHAIN QUESTION")
    print("====================================")

    data = {

        "selected_subject":
            "Supply Chain",

        "question":
            "What is reverse logistics?"

    }

    response = requests.post(

        f"{BASE_URL}/ask",

        json=data

    )

    print("Status:", response.status_code)

    print("Response:")
    print(response.json())


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("############################################")
    print("#     SUPPLY CHAIN AI SERVICE TEST         #")
    print("############################################")

    try:

        test_health()

        test_carbon_prediction()

        test_subject_question()

        test_off_topic_question()

        test_supply_chain_question()

        print("\n")
        print("############################################")
        print("#            TESTING COMPLETE              #")
        print("############################################")

    except requests.exceptions.ConnectionError:

        print("\nERROR:")
        print(
            "Could not connect to AI Service."
        )

        print("\nStart the server first:")

        print(
            "uvicorn AI_SERVICE.main:app "
            "--reload --port 8000"
        )