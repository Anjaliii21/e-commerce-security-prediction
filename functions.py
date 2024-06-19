import numpy as np
import pandas as pd
import pickle
from flask import render_template


def load_model():
    with open("model.pickle", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pickle", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_spam(email_content, model, vectorizer):
    data = pd.DataFrame({"Message": [email_content]})
    email_features = vectorizer.transform(data["Message"])
    prediction = model.predict(email_features)[0]
    return "Not Spam" if prediction == 1 else "Spam"


def load_credit_model():
    with open("credit_model.pickle", "rb") as f:
        credit_model = pickle.load(f)
    return credit_model


def predict_credit(card_content, credit_model):
    card_features = card_content.split(",")
    if len(card_features) != 30:
        return render_template("creditcard.html", credit_transaction_prediction=None,
                               error="Invalid number of features. Please provide 30 comma-separated values.")
    card_features = np.array(card_features, dtype=np.float64).reshape(1, -1)

    credit_prediction = credit_model.predict(card_features)[0]
    return "Fraudulent transaction" if credit_prediction == 1 else "Legitimate transaction"


import numpy as np
import pandas as pd
import pickle
from flask import render_template
import re


def load_model():
    with open("model.pickle", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pickle", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_spam(email_content, model, vectorizer):
    data = pd.DataFrame({"Message": [email_content]})
    email_features = vectorizer.transform(data["Message"])
    prediction = model.predict(email_features)[0]
    return "Not Spam" if prediction == 1 else "Spam"


def load_credit_model():
    with open("credit_model.pickle", "rb") as f:
        credit_model = pickle.load(f)
    return credit_model


def predict_credit(card_content, credit_model):
    card_features = card_content.split(",")
    if len(card_features) != 30:
        return render_template("creditcard.html", credit_transaction_prediction=None,
                               error="Invalid number of features. Please provide 30 comma-separated values.")
    card_features = np.array(card_features, dtype=np.float64).reshape(1, -1)

    credit_prediction = credit_model.predict(card_features)[0]
    return "Fraudulent transaction" if credit_prediction == 1 else "Legitimate transaction"



def check_password_strength(password):
    strength = 0

    # Check password length
    if len(password) < 6:
        strength = 0
    elif len(password) < 10:
        strength = 1
    else:
        strength = 2

    # Check for digits
    if re.search(r"\d", password):
        strength += 1

    # Check for uppercase
    if re.search(r"[A-Z]", password):
        strength += 1

    # Check for lowercase
    if re.search(r"[a-z]", password):
        strength += 1

    # Check for special characters
    if re.search(r"\W", password):
        strength += 1

    if strength < 2:
        return "Weak"
    elif strength < 4:
        return "Medium"
    else:
        return "Strong"
