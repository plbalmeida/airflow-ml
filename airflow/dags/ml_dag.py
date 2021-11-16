from airflow.models import DAG
from datetime import datetime, timedelta
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import os


default_args = {
    'owner': 'plbalmeida',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=0.25)
}

path = os.getcwd()

with DAG(
    dag_id='ml_pipeline',
    description='DAG for ML pipeline', 
    default_args=default_args, 
    start_date=days_ago(1),
    schedule_interval='@once',
    is_paused_upon_creation=True,
    ) as dag:
    
    get_data = BashOperator(
        task_id='get_data',
        #bash_command='python {}/src/data/make_dataset.py'.format(path),
        bash_command='python3 /home/ana/Documentos/airflowtest/src/data/make_dataset.py',
        retries=3,
        dag=dag
        )

    feature_engineer = BashOperator(
        task_id='feature_engineer',
        #bash_command='python {}/src/features/build_features.py'.format(path),
        bash_command='python3 /home/ana/Documentos/airflowtest/src/features/build_features.py',
        retries=3,
        dag=dag
        )

    train_model = BashOperator(
        task_id='train_model',
        #bash_command='python {}/src/models/train.py'.format(path),
        bash_command='python3 /home/ana/Documentos/airflowtest/src/models/train.py',
        retries=3,
        dag=dag
        )

    prediction = BashOperator(
        task_id='prediction',
        #bash_command='python {}/src/models/prediction.py'.format(path),
        bash_command='python3 /home/ana/Documentos/airflowtest/src/models/prediction.py',
        retries=3,
        dag=dag
        )

    get_data >> feature_engineer >> train_model >> prediction