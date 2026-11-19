
def calculate_risk(probability):

    if probability >= 0.85:
        return "Critical"

    elif probability >= 0.60:
        return "High"

    elif probability >= 0.30:
        return "Medium"

    else:
        return "Low"


def get_risk_details(probability):

    risk = calculate_risk(probability)

    recommendations = {

        "Critical": "Immediate machine inspection required.",

        "High": "Schedule urgent maintenance.",

        "Medium": "Plan preventive inspection.",

        "Low": "Continue routine monitoring."
    }

    return {
        "risk_level": risk,
        "recommendation": recommendations[risk]
    }
