import os
from pathlib import Path
import supervisely as sly


app_root_directory = str(Path(__file__).parents[1])
app_data_dir = os.path.join(app_root_directory, "tempfiles")
project_dir = os.path.join(app_data_dir, "project_dir")

models_data = sly.json.load_json_file("models/models.json")
if sly.is_production():
    app_session_id = sly.io.env.task_id()
else:
    app_session_id = 777

train_artifacts_dir = os.path.join(app_data_dir, "artifacts")
if not os.path.exists(train_artifacts_dir):
    os.mkdir(train_artifacts_dir)

stop_training = False
