import mlflow
import os
import sys

def verify():
    try:
        mlflow.set_tracking_uri('https://dagshub.com/akubima/Eksperimen_SML_Bima-Mukhlisin-Bil-Sajjad.mlflow')
        client = mlflow.tracking.MlflowClient()
        run = client.create_run(experiment_id='0', tags={'test': 'connection'})
        client.delete_run(run.info.run_id)
        print('SUCCESS')
    except Exception as e:
        print('FAILED')

if __name__ == '__main__':
    verify()
