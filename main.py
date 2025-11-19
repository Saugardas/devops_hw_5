import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow

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

# Дальше подготавливаем данные для эксперимента
iris_target_encoded, class_names = pd.factorize(iris_target) # Кодирование текстовых меток целевой переменной
scaler = StandardScaler()
iris_values_scaled = scaler.fit_transform(iris_values)
X_train, X_test, y_train, y_test = train_test_split(iris_values_scaled, iris_target_encoded, test_size=0.3)

# Возможные значения количества соседей
k_values = [3, 5, 10, 15]

for k in k_values:
    with mlflow.start_run(run_name=f"knn_for_k_{k}"): # названия разные чтобы разделять
        mlflow.log_param("n_neighbors", k) # Логируем гиперпараметр
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        y_pred = knn.predict(X_test)
        
        # метрики
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        
        # Логируем метрики
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Сохраняем модель
        mlflow.sklearn.log_model(knn, artifact_path=f"model_k{k}")
        
        # Выводим метрики
        print(f"Эксперимент с n_neighbors={k}: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}")
