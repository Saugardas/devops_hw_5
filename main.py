import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity

iris_data = pd.read_csv("./data/dataset.csv", sep=',')
print(iris_data.info())

def deepchecks_info(df, label):
    """
    Функция для проверки данных с помощью Deepchecks
    """
    dataset = Dataset(df, label=label, cat_features=[]) # в variety лежит наш целевой признак
    result = data_integrity().run(dataset)
    
    # Сохраняем отчет
    result.save_as_html('reports/deepchecks_report.html', as_widget=False, requirejs=False)

    print("Deepchecks отчет сохранен в reports/deepchecks_report.html")

deepchecks_info(iris_data, label='variety')
