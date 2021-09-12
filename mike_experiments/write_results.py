from sklearn.metrics import hamming_loss, accuracy_score, f1_score, precision_score, classification_report
from datetime import datetime
from configurations import config


def results_to_txt(model, y_test, predictions, df, vectorizer, runtime, grid_search_results):
    estimator = model.get_params()
    accuracy = accuracy_score(y_test, predictions)
    best_model = model.best_estimator_
    best_params = model.best_params_
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

#
# import win32com.client as win32
# outlook = win32.Dispatch('outlook.application')
# mail = outlook.CreateItem(0)
# mail.To = f'{config.mike_email}; {config.mike_phone}'
# mail.Subject = 'Message subject'
# mail.Body = 'Model is done training\n\n'
# # mail.HTMLBody = '<h2>Model is done training.</h2>' #this field is optional
# mail.Send()
# mail.close()