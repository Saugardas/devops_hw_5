import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def split_df(df):
    """
    Функция для разделения данных на обучающую и тестовую выборки
    """

    # Кодирование текстовых меток целевой переменной
    iris_target_encoded, _ = pd.factorize(iris_target)
    scaler = StandardScaler()
    iris_values_scaled = scaler.fit_transform(iris_values)
    return train_test_split(iris_values_scaled, iris_target_encoded, test_size=0.3)

def deepchecks_info(df, label):
    """
    Функция для проверки данных с помощью Deepchecks
    """
    dataset = Dataset(df, label=label, cat_features=[]) # в variety лежит наш целевой признак
    result = data_integrity().run(dataset)
    
    # Сохраняем отчет
    result.save_as_html('reports/deepchecks_report.html', as_widget=False, requirejs=False)

    print("Deepchecks отчет сохранен в reports/deepchecks_report.html")

def evidently_report(features):
    """Анализ дрейфа данных с EvidentlyAI"""
       
    ref, cur = train_test_split(features, test_size=0.3, random_state=42)
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    # Сохраняем отчет
    report.save_html('reports/evidently_report.html')
    print("EvidentlyAI отчет сохранен в reports/evidently_report.html")


iris_data = pd.read_csv("./data/dataset.csv", sep=',')
iris_values = iris_data[['sepal.length', 'sepal.width']]
iris_target = iris_data['variety']
# дальше составляем отчёты
deepchecks_info(iris_data, label='variety')
evidently_report(iris_values) # отправляем только признаки
