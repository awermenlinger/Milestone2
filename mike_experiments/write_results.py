from sklearn.metrics import hamming_loss, accuracy_score, f1_score, precision_score, classification_report
from datetime import datetime
from MyCreds.mycreds import Google
import smtplib


def send_email_report(txt_body):
    gmail_user = Google.email
    gmail_password = Google.appkey

    sent_from = Google.email
    to = [Google.email, Google.phone]
    txt_body = f"{txt_body}\n\n- {Google.email}"

    email_text = f"""From: {Google.email}\nTo: {", ".join(to)}\n\n{txt_body}"""

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, email_text)
        server.close()

        print('Email sent!')

    except:
        print('Something went wrong...')


def results_to_txt(model, y_test, predictions, df, vectorizer, runtime, grid_search_results):
    estimator = model.get_params()
    accuracy = accuracy_score(y_test, predictions)

    if grid_search_results:
        best_params = model.best_params_
        best_model = model.best_estimator_
    else:
        best_params = "No GridSearchCV"
        best_model = "No GridSearchCV"
    f1_weighted = f1_score(y_test, predictions, average='weighted')
    f1_micro = f1_score(y_test, predictions, average='micro')
    hammingloss = hamming_loss(y_test, predictions)
    precision_avg_samples = precision_score(y_test, predictions, average='samples')
    class_report = classification_report(y_test, predictions)

    txt_body = f"{estimator}\nRun Time: {runtime}\nDataframe Size: {df.shape}\nBest Model: {best_model}\n\
Best Params: {best_params}\nGridSearch Results: {grid_search_results}\n\nAccuracy: {accuracy}\n\
F1 Score (weighted): {f1_weighted}\nF1 Score (micro): {f1_micro}\nHamming Loss: {hammingloss}\n\
Precision (average by samples): {precision_avg_samples}\n\nClassification Report: \n{class_report}"

    print(txt_body)
    filepath = "results/"
    filename = f"{str(estimator['estimator'])[:-2]}_{vectorizer}_{datetime.now().strftime('%Y-%m-%d_%H%M')}.txt"

    with open(f"{filepath}{filename}", 'w') as file:
        file.write(txt_body)

    send_email_report(txt_body)
