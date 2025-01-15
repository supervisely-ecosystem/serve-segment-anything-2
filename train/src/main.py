import supervisely as sly
import os
from dotenv import load_dotenv
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    ClassesTable,
    Container,
    DoneLabel,
    Empty,
    Field,
    FileThumbnail,
    Image as SlyImage,
    Input,
    InputNumber,
    NotificationBox,
    Progress,
    RadioTable,
    RadioTabs,
    RandomSplitsTable,
    ReloadableArea,
    SelectDatasetTree,
    SelectString,
    Stepper,
    TaskLogs,
    Text,
    TrainValSplits,
    Tooltip,
    GridChart,
    FolderThumbnail,
    GridGallery,
)
import src.globals as g
from src.dataset_cache import download_project
import supervisely.io.env as env
import random
import numpy as np
from PIL import Image
import cv2
import torch
from src.utils import (
    get_multilabel_mask,
    resize_image_and_mask,
    generate_artificial_prompts,
    generate_prompts_for_batch,
    generate_history_df,
    generate_predictions,
    clip_gradients,
)
from train.src.dataset import SAM2Dataset
from train.src.data_sampler import BatchSampler
from functools import partial
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses import (
    FocalLoss,
    DiceLoss,
    JaccardLoss,
    LovaszLoss,
    MCCLoss,
    TverskyLoss,
)


# load credentials
load_dotenv("debug.env")
load_dotenv("supervisely.env")

# get api, team id and server address
api = sly.Api()
team_id = sly.env.team_id()
server_address = sly.env.server_address()


def update_split_tabs_for_nested_datasets(selected_dataset_ids):
    global dataset_ids, train_val_split, ds_name_to_id
    sum_items_count = 0
    temp_dataset_names = set()
    temp_dataset_infos = []
    datasets_tree = api.dataset.get_tree(project_id)

    dataset_id_to_info = {}
    ds_name_to_id = {}

    def _get_dataset_ids_infos_map(ds_tree):
        for ds_info in ds_tree.keys():
            dataset_id_to_info[ds_info.id] = ds_info
            if ds_tree[ds_info]:
                _get_dataset_ids_infos_map(ds_tree[ds_info])

    _get_dataset_ids_infos_map(datasets_tree)

    def _get_full_name(ds_id):
        ds_info = dataset_id_to_info[ds_id]
        full_name = ds_info.name
        while ds_info.parent_id is not None:
            ds_info = dataset_id_to_info[ds_info.parent_id]
            full_name = ds_info.name + "/" + full_name
        return full_name

    for ds_id in selected_dataset_ids:

        def _get_dataset_infos(ds_tree, nested=False):

            for ds_info in ds_tree.keys():
                need_add = ds_info.id == ds_id or nested
                if need_add:
                    temp_dataset_infos.append(ds_info)
                    name = _get_full_name(ds_info.id)
                    temp_dataset_names.add(name)
                    ds_name_to_id[name] = ds_info.id
                if ds_tree[ds_info]:
                    _get_dataset_infos(ds_tree[ds_info], nested=need_add)

        _get_dataset_infos(datasets_tree)

    dataset_ids = list(set([ds_info.id for ds_info in temp_dataset_infos]))
    unique_ds = set([ds_info for ds_info in temp_dataset_infos])
    sum_items_count = sum([ds_info.items_count for ds_info in unique_ds])

    contents = []
    split_methods = []
    tabs_descriptions = []

    split_methods.append("Random")
    tabs_descriptions.append("Shuffle data and split with defined probability")
    contents.append(
        Container([RandomSplitsTable(sum_items_count)], direction="vertical", gap=5)
    )

    split_methods.append("Based on item tags")
    tabs_descriptions.append("Images should have assigned train or val tag")
    contents.append(train_val_split._get_tags_content())

    split_methods.append("Based on datasets")
    tabs_descriptions.append("Select one or several datasets for every split")

    notification_box = NotificationBox(
        title="Notice: How to make equal splits",
        description="Choose the same dataset(s) for train/validation to make splits equal. Can be used for debug and for tiny projects",
        box_type="info",
    )
    train_ds_select = SelectString(temp_dataset_names, multiple=True)
    val_ds_select = SelectString(temp_dataset_names, multiple=True)
    train_val_split._train_ds_select = train_ds_select
    train_val_split._val_ds_select = val_ds_select
    train_field = Field(
        train_ds_select,
        title="Train dataset(s)",
        description="all images in selected dataset(s) are considered as training set",
    )
    val_field = Field(
        val_ds_select,
        title="Validation dataset(s)",
        description="all images in selected dataset(s) are considered as validation set",
    )

    contents.append(
        Container(
            widgets=[notification_box, train_field, val_field],
            direction="vertical",
            gap=5,
        )
    )
    content = RadioTabs(
        titles=split_methods,
        descriptions=tabs_descriptions,
        contents=contents,
    )
    train_val_split._content = content
    train_val_split.update_data()
    train_val_split_area.reload()


# function for updating global variables
def update_globals(new_dataset_ids):
    sly.logger.debug(f"Updating globals with new dataset_ids: {new_dataset_ids}")
    global dataset_ids, project_id, workspace_id, project_info, project_meta
    dataset_ids = new_dataset_ids
    if dataset_ids and all(ds_id is not None for ds_id in dataset_ids):
        project_id = api.dataset.get_info_by_id(dataset_ids[0]).project_id
        workspace_id = api.project.get_info_by_id(project_id).workspace_id
        project_info = api.project.get_info_by_id(project_id)
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        print(f"Project is {project_info.name}, {dataset_ids}")
    elif project_id:
        workspace_id = api.project.get_info_by_id(
            project_id, raise_error=True
        ).workspace_id
        project_info = api.project.get_info_by_id(project_id)
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    else:
        print("All globals set to None")
        dataset_ids = []
        project_id, workspace_id, project_info, project_meta = [None] * 4


# if app had started from context menu, one of this has to be set:
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)
dataset_ids = [dataset_id] if dataset_id else []
update_globals(dataset_ids)


### 1. Dataset selection
dataset_selector = SelectDatasetTree(
    project_id=project_id,
    multiselect=True,
    select_all_datasets=True,
    allowed_project_types=[sly.ProjectType.IMAGES],
)
use_cache_text = Text(
    "Use cached data stored on the agent to optimize project download"
)
use_cache_checkbox = Checkbox(use_cache_text, checked=True)
select_data_button = Button("Select data")
select_done = DoneLabel("Successfully selected input data")
select_done.hide()
reselect_data_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>Reselect data',
    button_type="warning",
    button_size="small",
    plain=True,
)
reselect_data_button.hide()
project_settings_content = Container(
    [
        dataset_selector,
        use_cache_checkbox,
        select_data_button,
        select_done,
        reselect_data_button,
    ]
)
card_project_settings = Card(
    title="Dataset selection", content=project_settings_content
)


### 2. Project classes
classes_table = ClassesTable(allowed_types=[sly.Bitmap, sly.Polygon])
select_classes_button = Button("select classes")
select_classes_button.hide()
select_other_classes_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>select other classes',
    button_type="warning",
    button_size="small",
    plain=True,
)
select_other_classes_button.hide()
classes_done = DoneLabel()
classes_done.hide()
classes_content = Container(
    [
        classes_table,
        select_classes_button,
        select_other_classes_button,
        classes_done,
    ]
)
card_classes = Card(
    title="Training classes",
    description=("Select classes, that should be used for training"),
    content=classes_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_classes.collapse()
card_classes.lock()


### 3. Train / validation split
train_val_split = TrainValSplits(project_id=project_id)
train_val_split_area = ReloadableArea(train_val_split)
split_data_button = Button("Split data")
resplit_data_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>Re-split data',
    button_type="warning",
    button_size="small",
    plain=True,
)
resplit_data_button.hide()
split_done = DoneLabel("Data was successfully splitted")
split_done.hide()
train_val_content = Container(
    [
        train_val_split_area,
        split_data_button,
        resplit_data_button,
        split_done,
    ]
)
card_train_val_split = Card(
    title="Train / validation split",
    description="Define how to split your data into train / val subsets",
    content=train_val_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_train_val_split.collapse()
card_train_val_split.lock()


### 4. Model selection
model_tabs_titles = ["Pretrained models", "Custom models"]
models_table_columns = [
    key for key in g.models_data[0].keys() if key not in ["weights_path", "config"]
]
models_table_rows = []
for element in g.models_data:
    models_table_rows.append(list(element.values())[:-2])
pretrained_models_table = RadioTable(
    columns=models_table_columns,
    rows=models_table_rows,
)
team_files_url = f"{env.server_address()}/files/"
team_files_button = Button(
    text="Open Team Files",
    button_type="info",
    plain=True,
    icon="zmdi zmdi-folder",
    link=team_files_url,
)
custom_model_path_input = Input(placeholder=f"Path to model file in Team Files")
custom_model_path_input_f = Field(
    custom_model_path_input,
    title=f"Copy path to model file from Team Files and paste to field below",
    description="Copy path in Team Files",
)
custom_model_file_thumbnail = FileThumbnail()
select_config = SelectString(
    values=[
        "sam2.1_hiera_t.yaml",
        "sam2.1_hiera_s.yaml",
        "sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_l.yaml",
    ]
)
select_config_f = Field(select_config, "Select model config")
custom_tab_content = Container(
    [
        team_files_button,
        custom_model_path_input_f,
        custom_model_file_thumbnail,
        select_config_f,
    ]
)
model_tabs = RadioTabs(
    titles=model_tabs_titles,
    contents=[pretrained_models_table, custom_tab_content],
)
select_model_button = Button("Select model")
reselect_model_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>Reselect model',
    button_type="warning",
    button_size="small",
    plain=True,
)
reselect_model_button.hide()
model_not_found_text = Text("Custom model not found", status="error")
model_not_found_text.hide()
model_select_done = DoneLabel("Model was successfully selected")
model_select_done.hide()
model_selection_content = Container(
    [
        model_tabs,
        select_model_button,
        reselect_model_button,
        model_not_found_text,
        model_select_done,
    ]
)
card_model_selection = Card(
    title="Model settings",
    description="Choose model size or how weights should be initialized",
    content=model_selection_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_model_selection.collapse()
card_model_selection.lock()


### 5. Training hyperparameters
n_epochs_input = InputNumber(value=100, min=1)
n_epochs_input_f = Field(
    content=n_epochs_input,
    title="Number of epochs",
    description=(
        "Number of epochs is equal to the number of times the model will see the entire dataset during training."
        " One epoch recflects a full training cycle through all images of the training dataset."
    ),
)
batch_size_input = InputNumber(value=2, min=1)
batch_size_input_f = Field(
    content=batch_size_input,
    title="Batch size",
    description=(
        "Batch size is the number of images that are being passed through the model during each training iteration."
        " Small batch sizes require less GPU memory, while large batch sizes can speed up the training process by "
        "reducing number of weights updates required within one epoch."
    ),
)
patience_input = InputNumber(value=20, min=1)
patience_input_f = Field(
    content=patience_input,
    title="Patience",
    description="Number of validation epochs to wait for no observable improvement for early stopping of training",
)
validation_freq_input = InputNumber(value=1, min=1)
validation_freq_input_f = Field(
    content=validation_freq_input,
    title="Validation and checkpoint save frequency",
    description="Validate model and save checkpoint every Nth epoch. Only best checkpoints will be saved",
)
learning_rate_input = InputNumber(value=1e-4, min=1e-8, max=0.1, step=0.00001)
learning_rate_input_f = Field(
    content=learning_rate_input,
    title="Learning rate",
    description=(
        "Learning rate is a parameter that defines the step size taken towards a global minimum of the "
        "loss function during model training. During each training iteration, learning rate is multiplied"
        " by the gradient of the loss function to update model weights. Too low learning rate can lead to"
        " slow convergence, while too high learning rate can cause the model to overshoot optimal weights."
    ),
)
use_scheduler_checkbox = Checkbox(content="enable", checked=True)
use_scheduler_checkbox_f = Field(
    content=use_scheduler_checkbox,
    title="Learning rate scheduler",
    description=(
        "Learning rate schedulers dynamically adjust the learning rate according to some specific "
        "schedule. Adaptive learning rate helps to achieve faster model convergence and better "
        "performance."
    ),
)
lr_scheduler_select = SelectString(
    values=[
        "StepLR",
        "LambdaLR",
        "MultiplicativeLR",
        "ReduceLROnPlateau",
    ],
    items_links=[
        "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR",
        "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR",
        "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR",
        "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau",
    ],
)
lr_scheduler_info = Text(
    text="Decays the learning rate of each parameter group by gamma every step_size epochs",
    status="info",
)
step_size_input = InputNumber(value=5, min=1)
step_size_input_f = Field(
    content=step_size_input,
    title="",
    description="step size",
)
gamma_input = InputNumber(value=0.5, step=0.1, min=1e-10, max=0.9)
gamma_input_f = Field(
    content=gamma_input,
    title="",
    description="gamma",
)
step_lr_container = Container(
    widgets=[step_size_input_f, gamma_input_f, Empty()],
    direction="horizontal",
    overflow="wrap",
    fractions=[1, 1, 5],
)
lambda_func_input = Input(value="lambda epoch: 0.95 ** epoch")
lambda_mult_container = Container(
    widgets=[lambda_func_input, Empty()],
    direction="horizontal",
    overflow="wrap",
    fractions=[1, 3],
)
lambda_mult_container_f = Field(
    content=lambda_mult_container, title="", description="Lambda function"
)
lambda_mult_container_f.hide()
factor_input = InputNumber(value=0.1, step=0.1, min=1e-10, max=0.9)
factor_input_f = Field(
    content=factor_input,
    title="",
    description="factor",
)
plateau_patience_input = InputNumber(value=10, step=1, min=1)
plateau_patience_input_f = Field(
    content=plateau_patience_input,
    title="",
    description="patience",
)
plateau_lr_container = Container(
    widgets=[factor_input_f, plateau_patience_input_f, Empty()],
    direction="horizontal",
    overflow="wrap",
    fractions=[1, 1, 5],
)
plateau_lr_container.hide()
weight_decay_input = InputNumber(value=1e-4, min=1e-8, max=0.1, step=0.00001)
weight_decay_input_f = Field(
    content=weight_decay_input,
    title="Weight decay",
    description=(
        "Weight decay is a technique used in machine learning in order to prevent overfitting. "
        "Weight decay helps to stabilize model training process by preventing model weights "
        "from having too large values. It is necessary to avoid having large values of model "
        "weights since it makes model sensitive to noise in data and worsens model's generalization "
        "capabilities."
    ),
)
clip_gradients_checkbox = Checkbox(content="enable", checked=True)
clip_gradients_checkbox_f = Field(
    content=clip_gradients_checkbox,
    title="Gradient clipping",
    description=(
        "Gradient clipping is a technique used in machine learning in order to prevent the gradients"
        " from having too large values during training process. This phenomenon is called exploding "
        "gradients problem and can lead to large shifts in model weights during training. Gradient "
        "clipping is used for maintaining numerical stability during training by limiting the values"
        " of the gradients."
    ),
)
clip_gradients_type_select = SelectString(
    values=["by norm", "by value"],
    items_links=[
        "https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html",
        "https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html",
    ],
)
clip_gradients_type_select_f = Field(
    content=clip_gradients_type_select,
    title="Gradient clipping method",
    description=(
        "Clipping by value assumes setting of maximum allowed value for gradients. If gradient "
        "exceeds this value, it is being clipped to this threshold. Clipping by norm assumes"
        " multiplying the unit vector of the gradients with given threshold."
    ),
)
clip_gradients_threshold = InputNumber(value=1.0, min=0.1, step=0.1)
clip_gradients_threshold_f = Field(
    content=clip_gradients_threshold,
    title="",
    description="clipping threshold",
)
grad_acc_checkbox = Checkbox(content="enable", checked=False)
grad_acc_checkbox_f = Field(
    content=grad_acc_checkbox,
    title="Gradient accumulation",
    description=(
        "Gradient accumulation is a technique used in machine learning in order to imitate a larger "
        "batch size. Lack of free GPU memory may force to use small batch sizes, but it can slow down"
        " model convergence. In order to overcome this issue, we can update the weights of our model "
        "every N batches instead of updating them every batch as usually. The gradients keep accumulating"
        " for each of batches untill N accumulation steps will not be reached - after that model's weights"
        " are being updated."
    ),
)
n_acc_steps_input = InputNumber(value=4, min=2)
n_acc_steps_input_f = Field(
    content=n_acc_steps_input,
    title="",
    description="number of accumulation steps",
)
n_acc_steps_input_f.hide()
loss_func_select = SelectString(
    values=[
        "Focal + Dice loss",
        "FocalLoss",
        "CrossEntropyLoss",
        "DiceLoss",
        "JaccardLoss",
        "TverskyLoss",
        "MCCLoss",
        "LovaszLoss",
    ]
)
loss_func_select_f = Field(
    content=loss_func_select,
    title="Segmentation loss function",
    description=(
        "Loss function is a mathematical function for estimating the difference between labels "
        "predicted by model and ground truth labels. The goal of the training process is to find"
        " the global minimum of the loss function at which model will perform best."
    ),
)
loss_func_info = Text(
    text=(
        "Original loss function used for SAM 2 training. A linear combination of Focal and Dice "
        "losses with a ratio of 20:1 respectively."
    ),
    status="info",
)
focal_gamma_input = InputNumber(value=2, min=0, step=0.1)
focal_gamma_input_f = Field(
    content=focal_gamma_input,
    title="",
    description="gamma (focusing parameter for adjusting the rate at which easy samples are down-weighted)",
)
focal_gamma_input_f.hide()
tversky_alpha_input = InputNumber(value=0.6, min=0.1, max=0.9, step=0.1)
tversky_alpha_input_f = Field(
    content=tversky_alpha_input,
    title="",
    description="alpha",
)
tversky_beta_input = InputNumber(value=0.4, min=0.1, max=0.9, step=0.1)
tversky_beta_input_f = Field(
    content=tversky_beta_input,
    title="",
    description="beta",
)
tversky_param_container = Container(
    widgets=[tversky_alpha_input_f, tversky_beta_input_f, Empty()],
    direction="horizontal",
    overflow="wrap",
    fractions=[1, 1, 5],
)
tversky_param_container.hide()
min_points_input = InputNumber(value=1, min=1)
min_points_input_f = Field(content=min_points_input, title="min")
max_points_input = InputNumber(value=3, min=1)
max_points_input_f = Field(content=max_points_input, title="max")
n_points_input = Container(
    widgets=[
        min_points_input_f,
        max_points_input_f,
        Empty(),
    ],
    direction="horizontal",
    overflow="wrap",
    fractions=[1, 1, 5],
)
n_points_input_f = Field(
    content=n_points_input,
    title="Number of points per object for prompt generation",
    description="Define how many points should be used to generate prompt for one object mask",
)
preview_prompt_button = Button("Preview prompt generation")
preview_image_widget = SlyImage()
preview_image_f = Field(content=preview_image_widget, title="input image")
preview_mask_widget = SlyImage()
preview_mask_f = Field(
    content=preview_mask_widget, title="binarized mask with point prompts"
)
previews_container = Container(
    widgets=[preview_image_f, preview_mask_f, Empty()],
    direction="horizontal",
    overflow="wrap",
    fractions=[1, 1, 1],
)
previews_container.hide()
save_train_params_button = Button("Save training hyperparameters")
change_train_params_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>Change training hyperparameters',
    button_type="warning",
    button_size="small",
    plain=True,
)
change_train_params_button.hide()
train_params_done = DoneLabel("Successfully saved training hyperparameters")
train_params_done.hide()
train_params_content = Container(
    [
        n_epochs_input_f,
        batch_size_input_f,
        patience_input_f,
        validation_freq_input_f,
        learning_rate_input_f,
        use_scheduler_checkbox_f,
        lr_scheduler_select,
        lr_scheduler_info,
        step_lr_container,
        lambda_mult_container_f,
        plateau_lr_container,
        weight_decay_input_f,
        clip_gradients_checkbox_f,
        clip_gradients_type_select_f,
        clip_gradients_threshold_f,
        grad_acc_checkbox_f,
        n_acc_steps_input_f,
        loss_func_select_f,
        loss_func_info,
        focal_gamma_input_f,
        tversky_param_container,
        n_points_input_f,
        preview_prompt_button,
        previews_container,
        save_train_params_button,
        change_train_params_button,
        train_params_done,
    ]
)
card_train_params = Card(
    title="Training hyperparameters",
    description="Define general settings and advanced configuration",
    content=train_params_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_train_params.collapse()
card_train_params.lock()


### 6. Training progress
start_training_button = Button("start training")
stop_training_button = Button(text="stop training", button_type="danger")
stop_training_tooltip = Tooltip(
    text="all training artifacts will be saved",
    content=stop_training_button,
    placement="right",
)
stop_training_tooltip.hide()
start_stop_container = Container(
    widgets=[
        start_training_button,
        stop_training_tooltip,
        Empty(),
    ],
    direction="horizontal",
    overflow="wrap",
    fractions=[1, 1, 4],
)
logs_button = Button(
    text="Show logs",
    plain=True,
    button_size="mini",
    icon="zmdi zmdi-caret-down-circle",
)
task_logs = TaskLogs(task_id=g.app_session_id)
task_logs.hide()
progress_bar_download_project = Progress()
progress_bar_download_model = Progress()
progress_bar_epochs = Progress()
progress_bar_batches = Progress(hide_on_finish=False)
train_chart_titles = ["train loss", "train IoU"]
train_metric_charts = GridChart(data=train_chart_titles, columns=2, gap=40)
train_metric_charts_f = Field(train_metric_charts, "Train & validation metrics")
train_metric_charts_f.hide()
val_chart_titles = ["validation loss", "validation IoU"]
val_metric_charts = GridChart(data=val_chart_titles, columns=2, gap=20)
val_metric_charts.hide()
early_stopping_warning = Text()
early_stopping_warning.hide()
progress_bar_predictions = Progress()
predictions_gallery = GridGallery(
    columns_number=4,
    show_opacity_slider=False,
    enable_zoom=True,
    sync_views=True,
)
predictions_gallery_f = Field(predictions_gallery, "Model predictions visualization")
predictions_gallery_f.hide()
progress_bar_upload_artifacts = Progress()
train_done = DoneLabel(
    "Training completed. Training artifacts were uploaded to Team Files"
)
train_done.hide()
train_progress_content = Container(
    [
        start_stop_container,
        logs_button,
        task_logs,
        progress_bar_download_project,
        progress_bar_download_model,
        progress_bar_epochs,
        progress_bar_batches,
        train_metric_charts_f,
        val_metric_charts,
        early_stopping_warning,
        progress_bar_predictions,
        predictions_gallery_f,
        progress_bar_upload_artifacts,
        train_done,
    ],
)
card_train_progress = Card(
    title="Training progress",
    description="Track progress, detailed logs, metrics charts and other visualizations",
    content=train_progress_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_train_progress.collapse()
card_train_progress.lock()


### 7. Training artifacts
train_artifacts_folder = FolderThumbnail()
card_train_artifacts = Card(
    title="Training artifacts",
    description="Fine-tuned model weights, training history and predictions visualizations",
    content=train_artifacts_folder,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_train_artifacts.collapse()
card_train_artifacts.lock()


stepper = Stepper(
    widgets=[
        card_project_settings,
        card_classes,
        card_train_val_split,
        card_model_selection,
        card_train_params,
        card_train_progress,
        card_train_artifacts,
    ]
)


app = sly.Application(
    layout=Container(
        widgets=[
            stepper,
        ]
    ),
)


@dataset_selector.value_changed
def on_dataset_selected(new_dataset_ids):
    sly.logger.debug(f"Selected datasets widget value changed to: {new_dataset_ids}")
    if new_dataset_ids == []:
        select_data_button.hide()
    elif new_dataset_ids != [] and reselect_data_button.is_hidden():
        select_data_button.show()
    update_globals(new_dataset_ids)
    if sly.project.download.is_cached(project_id):
        use_cache_text.text = (
            "Use cached data stored on the agent to optimize project download"
        )
    else:
        use_cache_text.text = (
            "Cache data on the agent to optimize project download for future trainings"
        )


def _update_select_classes_button(selected_classes):
    n_classes = len(selected_classes)
    if n_classes > 0:
        if n_classes > 1:
            select_classes_button.text = f"Select {n_classes} classes"
        else:
            select_classes_button.text = f"Select {n_classes} class"
        select_classes_button.show()
    else:
        select_classes_button.hide()


@select_data_button.click
def select_input_data():
    selected_datasets = set()
    for dataset_id in dataset_selector.get_selected_ids():
        selected_datasets.add(dataset_id)
        for ds in api.dataset.get_nested(project_id=project_id, dataset_id=dataset_id):
            selected_datasets.add(ds.id)
    update_globals(list(selected_datasets))
    update_split_tabs_for_nested_datasets(dataset_ids)
    sly.logger.debug(f"Select data button clicked, selected datasets: {dataset_ids}")
    select_data_button.loading = True
    dataset_selector.disable()
    use_cache_text.disable()
    use_cache_checkbox.disable()
    classes_table.read_project_from_id(project_id, dataset_ids=dataset_ids)
    classes_table.select_all()
    selected_classes = classes_table.get_selected_classes()
    _update_select_classes_button(selected_classes)
    select_data_button.loading = False
    select_data_button.hide()
    select_done.show()
    reselect_data_button.show()
    curr_step = stepper.get_active_step()
    curr_step += 1
    stepper.set_active_step(curr_step)
    card_classes.unlock()
    card_classes.uncollapse()


@classes_table.value_changed
def on_classes_selected(selected_classes):
    _update_select_classes_button(selected_classes)


@reselect_data_button.click
def reselect_input_data():
    select_data_button.show()
    reselect_data_button.hide()
    select_done.hide()
    dataset_selector.enable()
    use_cache_text.enable()
    use_cache_checkbox.enable()
    curr_step = stepper.get_active_step()
    curr_step -= 1
    stepper.set_active_step(curr_step)


@select_classes_button.click
def select_classes():
    n_classes = len(classes_table.get_selected_classes())
    if n_classes > 1:
        classes_done.text = f"{n_classes} classes were selected successfully"
    else:
        classes_done.text = f"{n_classes} class was selected successfully"
    select_classes_button.hide()
    classes_done.show()
    select_other_classes_button.show()
    classes_table.disable()
    curr_step = stepper.get_active_step()
    curr_step += 1
    stepper.set_active_step(curr_step)
    card_train_val_split.unlock()
    card_train_val_split.uncollapse()


@select_other_classes_button.click
def select_other_classes():
    classes_table.enable()
    select_other_classes_button.hide()
    classes_done.hide()
    select_classes_button.show()
    curr_step = stepper.get_active_step()
    curr_step -= 1
    stepper.set_active_step(curr_step)


@split_data_button.click
def split_data():
    split_data_button.loading = True
    train_val_split.disable()
    split_done.show()
    split_data_button.loading = False
    split_data_button.hide()

    resplit_data_button.show()
    curr_step = stepper.get_active_step()
    curr_step += 1
    stepper.set_active_step(curr_step)
    card_model_selection.unlock()
    card_model_selection.uncollapse()


@resplit_data_button.click
def resplit_data():
    train_val_split.enable()
    split_data_button.show()
    split_done.hide()
    resplit_data_button.hide()
    curr_step = stepper.get_active_step()
    curr_step -= 1
    stepper.set_active_step(curr_step)


@model_tabs.value_changed
def model_tab_changed(value):
    if value == "Pretrained models":
        model_not_found_text.hide()
        model_select_done.hide()


@select_model_button.click
def select_model():
    weights_type = model_tabs.get_active_tab()
    file_exists = True
    if weights_type == "Custom models":
        custom_link = custom_model_path_input.get_value()
        if custom_link != "":
            file_exists = api.file.exists(sly.env.team_id(), custom_link)
        else:
            file_exists = False
    if not file_exists and weights_type == "Custom models":
        model_not_found_text.show()
        model_select_done.hide()
    else:
        model_select_done.show()
        model_not_found_text.hide()
        select_model_button.hide()
        model_tabs.disable()
        pretrained_models_table.disable()
        custom_model_path_input.disable()
        reselect_model_button.show()
        curr_step = stepper.get_active_step()
        curr_step += 1
        stepper.set_active_step(curr_step)
        card_train_params.unlock()
        card_train_params.uncollapse()


@reselect_model_button.click
def reselect_model():
    select_model_button.show()
    model_not_found_text.hide()
    model_select_done.hide()
    model_tabs.enable()
    pretrained_models_table.enable()
    custom_model_path_input.enable()
    reselect_model_button.hide()
    curr_step = stepper.get_active_step()
    curr_step -= 1
    stepper.set_active_step(curr_step)


@custom_model_path_input.value_changed
def change_file_preview(value):
    file_info = None
    if value != "":
        file_info = api.file.get_info_by_path(sly.env.team_id(), value)
    if file_info is None:
        model_not_found_text.show()
        model_select_done.hide()
        custom_model_file_thumbnail.set(None)
    else:
        model_not_found_text.hide()
        custom_model_file_thumbnail.set(file_info)


def change_lr_ui(lr_scheduler):
    if lr_scheduler == "StepLR":
        lr_scheduler_info.set(
            text="Decays the learning rate of each parameter group by gamma every step_size epochs",
            status="info",
        )
        step_lr_container.show()
        lambda_mult_container_f.hide()
        plateau_lr_container.hide()
    elif lr_scheduler == "LambdaLR":
        lr_scheduler_info.set(
            text="The learning rate of each parameter group is set to the initial learning rate times a given lambda function",
            status="info",
        )
        step_lr_container.hide()
        lambda_mult_container_f.show()
        plateau_lr_container.hide()
    elif lr_scheduler == "MultiplicativeLR":
        lr_scheduler_info.set(
            text="Multiplies the learning rate of each parameter group by the factor given in the lambda function",
            status="info",
        )
        step_lr_container.hide()
        lambda_mult_container_f.show()
        plateau_lr_container.hide()
    elif lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler_info.set(
            text=(
                "Reduces learning rate when a metric has stopped improving. Models often benefit from reducing "
                "the learning rate by a factor of 2-10 once learning stagnates. This scheduler reads a metrics"
                " quantity and if no improvement is seen for a patience number of epochs, the learning rate is"
                " reduced."
            ),
            status="info",
        )
        step_lr_container.hide()
        lambda_mult_container_f.hide()
        plateau_lr_container.show()


@use_scheduler_checkbox.value_changed
def change_lr_content_visibility(value):
    if value:
        lr_scheduler_select.show()
        lr_scheduler_info.show()
        lr_scheduler = lr_scheduler_select.get_value()
        change_lr_ui(lr_scheduler)
    else:
        lr_scheduler_select.hide()
        lr_scheduler_info.hide()
        step_lr_container.hide()
        lambda_mult_container_f.hide()
        plateau_lr_container.hide()


@lr_scheduler_select.value_changed
def change_lr_scheduler(value):
    change_lr_ui(value)


@clip_gradients_checkbox.value_changed
def show_gradient_clipping_content(value):
    if value:
        clip_gradients_type_select_f.show()
        clip_gradients_threshold_f.show()
    else:
        clip_gradients_type_select_f.hide()
        clip_gradients_threshold_f.hide()


@grad_acc_checkbox.value_changed
def show_gradient_accumulation_content(value):
    if value:
        n_acc_steps_input_f.show()
    else:
        n_acc_steps_input_f.hide()


@loss_func_select.value_changed
def change_loss_func(value):
    if value == "Focal + Dice loss":
        loss_func_info.set(
            text=(
                "Original loss function used for SAM 2 training. A linear combination of Focal and Dice "
                "losses with a ratio of 20:1 respectively."
            ),
            status="info",
        )
        focal_gamma_input_f.hide()
        tversky_param_container.hide()
    if value == "FocalLoss":
        loss_func_info.set(
            text=(
                "Applies a scaling factor to the cross entropy loss in order to focus learning on hard"
                " misclassified examples. This scaling factor decreases as model confidence in correct"
                " class increases, thereby down-weighting the contribution of easy samples, making model"
                " focus on hard samples during training. Works best with highly imbalanced datasets."
            ),
            status="info",
        )
        focal_gamma_input_f.show()
        tversky_param_container.hide()
    elif value == "CrossEntropyLoss":
        loss_func_info.set(
            text=(
                "Measures the difference between probability distributions of predicted and ground "
                "truth labels. Standard choice when training dataset has no significant class imbalance."
            ),
            status="info",
        )
        focal_gamma_input_f.hide()
        tversky_param_container.hide()
    elif value == "DiceLoss":
        loss_func_info.set(
            text=(
                "Differentiable modification of Dice coefficient - a metric used to evaluate the overlap between predicted"
                " and ground truth segmentation masks. Prevents the model from ignoring the minority classes by focusing "
                "on the overlapping regions between the predicted and ground truth mask."
            ),
            status="info",
        )
        focal_gamma_input_f.hide()
        tversky_param_container.hide()
    elif value == "JaccardLoss":
        loss_func_info.set(
            text=(
                "Differentiable modification of IoU (Jaccard index) - a gold standard metric used to evaluate performance"
                " of image segmentation models. Useful for cases where the balance between precision and recall is crucial."
            ),
            status="info",
        )
    elif value == "LovaszLoss":
        loss_func_info.set(
            text=(
                "A loss function for direct optimization of IoU score. Unlike many other losses, it operates directly on "
                "the IoU scores between the predicted and actual labels, rather than summing over pixel-wise errors. "
            ),
            status="info",
        )
        focal_gamma_input_f.hide()
        tversky_param_container.hide()
    elif value == "MCCLoss":
        loss_func_info.set(
            text=(
                "Based on Matthews correlation coefficient - a metric indicating the correlation between predicted and "
                "ground truth labels. Originally designed for medical image segmentation scenarios, MCC loss penalizes "
                "for both foreground and background pixel misclassifications. Efficient in scenarios with skewed class "
                "distributions."
            ),
            status="info",
        )
        focal_gamma_input_f.hide()
        tversky_param_container.hide()
    elif value == "TverskyLoss":
        loss_func_info.set(
            text=(
                "Modification of Dice loss which allows to control the penalties for false positive and false negative "
                "types of errors with the help of alpha and beta hyperparameters, respectively. These hyperparameters "
                "allow the tuning of the loss function to focus more on either false positives or false negatives during"
                "  training. This capability is useful in cases where the costs of different types of segmentation errors"
                " are not equal."
            ),
            status="info",
        )
        focal_gamma_input_f.hide()
        tversky_param_container.show()


@preview_prompt_button.click
def preview_prompt():
    # define directory for storing preview files
    preview_dir = os.path.join(g.app_data_dir, "preview")
    if not os.path.exists(preview_dir):
        os.mkdir(preview_dir)
    # find images which have labels
    for dataset_id in dataset_ids:
        preview_image_infos = api.image.get_list(
            dataset_id=dataset_id,
            filters=[
                {
                    "field": "labelsCount",
                    "operator": ">",
                    "value": "0",
                }
            ],
        )
        if len(preview_image_infos) > 0:
            break
    # select random image for preview
    preview_image_id = random.choice(preview_image_infos).id
    # download preview image
    preview_image_np = api.image.download_np(preview_image_id)
    # download preview image annotation
    ann_json = api.annotation.download(preview_image_id).annotation
    ann = sly.Annotation.from_json(ann_json, project_meta)
    # get points range and mask annotations
    min_points = min_points_input.get_value()
    max_points = max_points_input.get_value()
    # generate multilabel mask
    multilabel_mask = get_multilabel_mask(preview_image_np, ann)
    # resize input image and multilabel mask
    preview_image_np, multilabel_mask = resize_image_and_mask(
        image=preview_image_np,
        mask=multilabel_mask,
    )
    # generate prompts
    n_points = random.randint(min_points, max_points)
    prompts = generate_artificial_prompts(multilabel_mask, preview_image_np, n_points)
    bitmap = multilabel_mask.copy()
    bitmap = np.where(bitmap != 0, 255, 0).astype(np.uint8)
    bitmap = np.repeat(bitmap[..., np.newaxis], 3, axis=2)
    for point in prompts["point_coordinates"]:
        # draw points
        cv2.circle(
            img=bitmap,
            center=(point[0], point[1]),
            radius=6,
            color=(255, 0, 0),
            thickness=-1,
        )
    # save preview image and mask to local storage
    local_preview_image_path = os.path.join(preview_dir, "image.jpg")
    preview_image = Image.fromarray(preview_image_np)
    preview_image.save(local_preview_image_path)
    local_preview_mask_path = os.path.join(preview_dir, "mask.jpg")
    bitmap = Image.fromarray(bitmap)
    bitmap.save(local_preview_mask_path)
    # upload preview image and mask to Team Files
    remote_preview_image_path = os.path.join(
        sly.output.RECOMMENDED_EXPORT_PATH,
        sly.app.fastapi.get_name_from_env(),
        str(g.app_session_id),
        "image.jpg",
    )
    remote_preview_mask_path = os.path.join(
        sly.output.RECOMMENDED_EXPORT_PATH,
        sly.app.fastapi.get_name_from_env(),
        str(g.app_session_id),
        "mask.jpg",
    )
    preview_image_info = api.file.upload(
        team_id, local_preview_image_path, remote_preview_image_path
    )
    preview_mask_info = api.file.upload(
        team_id, local_preview_mask_path, remote_preview_mask_path
    )
    # show preview
    if sly.is_production():
        preview_image_widget.set(preview_image_info.storage_path)
        preview_mask_widget.set(preview_mask_info.storage_path)
    else:
        preview_image_widget.set(preview_image_info.full_storage_url)
        preview_mask_widget.set(preview_mask_info.full_storage_url)
    previews_container.show()


@save_train_params_button.click
def save_train_params():
    save_train_params_button.hide()
    train_params_done.show()
    change_train_params_button.show()
    n_epochs_input.disable()
    batch_size_input.disable()
    patience_input.disable()
    validation_freq_input.disable()
    learning_rate_input.disable()
    use_scheduler_checkbox.disable()
    lr_scheduler_select.disable()
    lr_scheduler_info.disable()
    step_size_input.disable()
    gamma_input.disable()
    lambda_func_input.disable()
    factor_input.disable()
    plateau_patience_input.disable()
    weight_decay_input.disable()
    clip_gradients_checkbox.disable()
    clip_gradients_type_select.disable()
    clip_gradients_threshold.disable()
    grad_acc_checkbox.disable()
    n_acc_steps_input.disable()
    loss_func_select.disable()
    focal_gamma_input.disable()
    tversky_alpha_input.disable()
    tversky_beta_input.disable()
    min_points_input.disable()
    max_points_input.disable()
    preview_prompt_button.disable()
    curr_step = stepper.get_active_step()
    curr_step += 1
    stepper.set_active_step(curr_step)
    card_train_progress.unlock()
    card_train_progress.uncollapse()


@change_train_params_button.click
def change_train_params():
    save_train_params_button.show()
    train_params_done.hide()
    change_train_params_button.hide()
    n_epochs_input.enable()
    batch_size_input.enable()
    patience_input.enable()
    validation_freq_input.enable()
    learning_rate_input.enable()
    lr_scheduler_select.enable()
    lr_scheduler_info.enable()
    step_size_input.enable()
    gamma_input.enable()
    lambda_func_input.enable()
    factor_input.enable()
    plateau_patience_input.enable()
    weight_decay_input.enable()
    clip_gradients_checkbox.enable()
    clip_gradients_type_select.enable()
    clip_gradients_threshold.enable()
    grad_acc_checkbox.enable()
    n_acc_steps_input.enable()
    loss_func_select.enable()
    focal_gamma_input.enable()
    tversky_alpha_input.enable()
    tversky_beta_input.enable()
    min_points_input.enable()
    max_points_input.enable()
    preview_prompt_button.enable()
    curr_step = stepper.get_active_step()
    curr_step -= 1
    stepper.set_active_step(curr_step)


@logs_button.click
def change_logs_visibility():
    if task_logs.is_hidden():
        task_logs.show()
        logs_button.text = "Hide logs"
        logs_button.icon = "zmdi zmdi-caret-up-circle"
    else:
        task_logs.hide()
        logs_button.text = "Show logs"
        logs_button.icon = "zmdi zmdi-caret-down-circle"


@stop_training_button.click
def stop_training_process():
    stop_training_button.loading = True
    g.stop_training = True


@start_training_button.click
def start_training():
    reselect_data_button.disable()
    select_other_classes_button.disable()
    resplit_data_button.disable()
    reselect_model_button.disable()
    change_train_params_button.disable()
    start_training_button.loading = True
    # download project to local storage
    use_cache = use_cache_checkbox.is_checked()
    dataset_infos = [
        api.dataset.get_info_by_id(dataset_id) for dataset_id in dataset_ids
    ]
    download_project(
        api=api,
        project_info=project_info,
        dataset_infos=dataset_infos,
        use_cache=use_cache,
        progress=progress_bar_download_project,
    )
    # remove unlabeled images
    n_images = sum([info.images_count for info in dataset_infos])
    n_images_before = n_images
    sly.Project.remove_items_without_objects(g.project_dir, inplace=True)
    # remove unselected classes
    selected_classes = classes_table.get_selected_classes()
    try:
        sly.Project.remove_classes_except(
            g.project_dir, classes_to_keep=selected_classes, inplace=True
        )
    except Exception:
        if not use_cache:
            raise
        sly.logger.warn(
            f"Error during classes removing. Will try to re-download project without cache",
            exc_info=True,
        )
        download_project(
            api=api,
            project_info=project_info,
            dataset_infos=dataset_infos,
            use_cache=False,
            progress=progress_bar_download_project,
        )
        sly.Project.remove_classes_except(
            g.project_dir, classes_to_keep=selected_classes, inplace=True
        )
    # validate splits
    project = sly.Project(g.project_dir, sly.OpenMode.READ)
    n_images_after = project.total_items
    if n_images_before != n_images_after:
        train_val_split._project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
        train_set, val_set = train_val_split.get_splits()
        val_part = len(val_set) / (len(train_set) + len(val_set))
        new_val_count = round(n_images_after * val_part)
        if new_val_count < 1:
            sly.app.show_dialog(
                title="An error occured",
                description="Val split length is 0 after ignoring images. Please check your data",
                status="error",
            )
            raise ValueError(
                "Val split length is 0 after ignoring images. Please check your data"
            )
    # split the data
    try:
        train_val_split._project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
        train_val_split._project_id = None
        train_val_split.update_data()
        train_set, val_set = train_val_split.get_splits()
        train_val_split._project_id = project_id
    except Exception:
        if not use_cache:
            raise
        sly.logger.warning(
            "Error during data splitting. Will try to re-download project without cache",
            exc_info=True,
        )
        download_project(
            api=api,
            project_info=project_info,
            dataset_infos=dataset_infos,
            use_cache=False,
            progress=progress_bar_download_project,
        )
        train_val_split._project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
        train_val_split._project_id = None
        train_val_split.update_data()
        train_set, val_set = train_val_split.get_splits()
        train_val_split._project_id = project_id
    # verify train and val set
    if len(train_set) == 0:
        raise ValueError("Train set is empty, check or change split configuration")
    if len(val_set) == 0:
        raise ValueError("Val set is empty, check or change split configuration")
    # load train and val datasets in sam 2 format
    train_dataset = SAM2Dataset(
        sly_items=train_set,
        sly_project_meta=project_meta,
    )
    val_dataset = SAM2Dataset(
        sly_items=val_set,
        sly_project_meta=project_meta,
    )
    # load model
    weights_type = model_tabs.get_active_tab()

    def download_monitor(monitor, api: sly.Api, progress: sly.Progress):
        value = monitor
        if progress.total == 0:
            progress.set(value, monitor.len, report=False)
        else:
            progress.set_current_value(value, report=False)
        weights_pbar.update(progress.current)

    if weights_type == "Pretrained models":
        selected_index = pretrained_models_table.get_selected_row_index()
        selected_dict = g.models_data[selected_index]
        weights_dst_path = selected_dict["weights_path"]
        if not sly.is_production():
            weights_dst_path = "." + weights_dst_path
        config_path = selected_dict["config"]
    elif weights_type == "Custom models":
        custom_link = custom_model_path_input.get_value()
        model_filename = "custom_model.pt"
        weights_dst_path = os.path.join(g.app_data_dir, model_filename)
        file_info = api.file.get_info_by_path(sly.env.team_id(), custom_link)
        if file_info is None:
            raise FileNotFoundError(f"Custon model file not found: {custom_link}")
        file_size = file_info.sizeb
        progress = sly.Progress(
            message="",
            total_cnt=file_size,
            is_size=True,
        )
        progress_cb = partial(download_monitor, api=api, progress=progress)
        with progress_bar_download_model(
            message="Downloading model weights...",
            total=file_size,
            unit="bytes",
            unit_scale=True,
        ) as weights_pbar:
            api.file.download(
                team_id=sly.env.team_id(),
                remote_path=custom_link,
                local_save_path=weights_dst_path,
                progress_cb=progress_cb,
            )
        config = select_config.get_value()
        config_path = "configs/sam2.1/" + config

    device = torch.device("cuda")
    sam2_model = build_sam2(config_path, weights_dst_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    # prepare data samplers and data loaders
    batch_size = batch_size_input.get_value()
    train_sampler = BatchSampler(train_set, project_meta, batch_size)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_sampler=train_sampler)
    val_sampler = BatchSampler(val_set, project_meta, batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_sampler=val_sampler)
    # set mask decoder and prompt encoder to train mode
    sam2_model.sam_mask_decoder.train(True)
    sam2_model.sam_prompt_encoder.train(True)
    sam2_model.image_encoder.train(False)
    # get training hyperparameters
    learning_rate = learning_rate_input.get_value()
    weight_decay = weight_decay_input.get_value()
    optimizer = torch.optim.AdamW(
        params=predictor.model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scaler = torch.cuda.amp.GradScaler()
    n_epochs = n_epochs_input.get_value()
    val_freq = validation_freq_input.get_value()
    patience = patience_input.get_value()
    min_points = min_points_input.get_value()
    max_points = max_points_input.get_value()
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    checkpoint_save_path = os.path.join(g.train_artifacts_dir, "fine-tuned_sam2.pt")
    do_grad_clip = clip_gradients_checkbox.is_checked()
    grad_clip_method = clip_gradients_type_select.get_value()
    grad_clip_thresh = clip_gradients_threshold.get_value()
    if grad_acc_checkbox.is_checked():
        n_acc_steps = n_acc_steps_input.get_value()
    else:
        n_acc_steps = 1
    use_scheduler = use_scheduler_checkbox.is_checked()
    if use_scheduler:
        scheduler_name = lr_scheduler_select.get_value()
        if scheduler_name == "StepLR":
            step_size = step_size_input.get_value()
            gamma = gamma_input.get_value()
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
        elif scheduler_name == "LambdaLR":
            lambda_func = eval(lambda_func_input.get_value())
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_func)
        elif scheduler_name == "MultiplicativeLR":
            lambda_func = eval(lambda_func_input.get_value())
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
                optimizer, lambda_func
            )
        elif scheduler_name == "ReduceLROnPlateau":
            factor = factor_input.get_value()
            plateau_patience = plateau_patience_input.get_value()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=factor, patience=plateau_patience
            )

    loss_name = loss_func_select.get_value()
    if loss_name == "Focal + Dice loss":

        class SAM2_original_loss:
            def __init__(self):
                self.focal_loss = FocalLoss(mode="binary")
                self.dice_loss = DiceLoss(mode="binary", from_logits=False)

            def __call__(self, prd_mask, gt_mask):
                focal_loss_score = self.focal_loss(prd_mask, gt_mask)
                dice_loss_score = self.dice_loss(prd_mask, gt_mask)
                score = focal_loss_score + 0.05 * dice_loss_score
                return score

        loss_func = SAM2_original_loss()

    elif loss_name == "FocalLoss":
        focal_gamma = focal_gamma_input.get_value()
        loss_func = FocalLoss(mode="binary", gamma=focal_gamma)
    elif loss_name == "CrossEntropyLoss":
        loss_func = torch.nn.BCELoss(reduction="mean")
    elif loss_name == "DiceLoss":
        loss_func = DiceLoss(mode="binary", from_logits=False)
    elif loss_name == "JaccardLoss":
        loss_func = JaccardLoss(mode="binary", from_logits=False)
    elif loss_name == "LovaszLoss":
        loss_func = LovaszLoss(mode="binary", from_logits=False)
    elif loss_name == "MCCLoss":
        loss_func = MCCLoss()
    elif loss_name == "TverskyLoss":
        tversky_alpha = tversky_alpha_input.get_value()
        tversky_beta = tversky_beta_input.get_value()
        loss_func = TverskyLoss(
            mode="binary", from_logits=False, alpha=tversky_alpha, beta=tversky_beta
        )

    # train loop
    with progress_bar_epochs(message="Epochs:", total=n_epochs) as epoch_pbar:
        best_val_iou = 0
        best_val_epoch = 0
        train_loss_history, train_iou_history = [], []
        val_loss_history, val_iou_history = [], []
        prev_val_epoch_iou = 0
        no_improvements = 0

        for epoch in range(1, n_epochs + 1):
            if g.stop_training:
                sly.logger.info(
                    "Training has been stopped early (by user or by reaching patience limit)"
                )
                break
            train_epoch_loss = 0
            train_epoch_iou = 0
            with progress_bar_batches(
                message="Training batches:", total=n_train_batches
            ) as batch_pbar:
                for batch_number, batch in enumerate(train_loader):
                    if g.stop_training:
                        break
                    n_points = random.randint(min_points, max_points)
                    images, masks = batch
                    # generate prompts for given batch
                    batch_images, batch_masks, batch_points, batch_labels = (
                        generate_prompts_for_batch(masks, images, n_points)
                    )

                    with torch.cuda.amp.autocast():
                        predictor.set_image_batch(batch_images)
                        mask_input, point_coordinates, point_labels, boxes = (
                            predictor._prep_prompts(
                                np.array(batch_points),
                                np.array(batch_labels),
                                box=None,
                                mask_logits=None,
                                normalize_coords=True,
                            )
                        )
                        sparse_embeddings, dense_embeddings = (
                            predictor.model.sam_prompt_encoder(
                                points=(point_coordinates, point_labels),
                                boxes=None,
                                masks=None,
                            )
                        )
                        high_res_features = [
                            feat_level[-1].unsqueeze(0)
                            for feat_level in predictor._features["high_res_feats"]
                        ]
                        low_res_masks, prd_scores, _, _ = (
                            predictor.model.sam_mask_decoder(
                                image_embeddings=predictor._features["image_embed"],
                                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=True,
                                repeat_image=False,
                                high_res_features=high_res_features,
                            )
                        )
                        prd_masks = predictor._transforms.postprocess_masks(
                            low_res_masks, predictor._orig_hw[-1]
                        )
                    gt_mask = torch.tensor(
                        np.array(batch_masks).astype(np.float32)
                    ).cuda()
                    prd_mask = torch.sigmoid(prd_masks[:, 0])
                    inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                    iou = inter / (
                        gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter
                    )
                    iou = np.mean(iou.cpu().detach().numpy())

                    if loss_name != "CrossEntropyLoss":
                        gt_mask = gt_mask.type(torch.int64)

                    seg_loss = loss_func(prd_mask, gt_mask)
                    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                    loss = seg_loss + score_loss * 0.05

                    train_epoch_loss += loss
                    train_epoch_iou += iou

                    # Backpropagation
                    loss = loss / n_acc_steps
                    scaler.scale(loss).backward()

                    if do_grad_clip:
                        clip_gradients(predictor, grad_clip_method, grad_clip_thresh)

                    if ((batch_number + 1) % n_acc_steps == 0) or (
                        (batch_number + 1) % n_train_batches == 0
                    ):
                        scaler.step(optimizer)
                        scaler.update()  # Update mixed precision
                        predictor.model.zero_grad()

                    batch_pbar.update()

                    if epoch == 1 and batch_number == 0:
                        stop_training_tooltip.show()

                    if (batch_number + 1) % n_train_batches == 0:
                        batch_pbar.reset()

            if not g.stop_training:

                train_epoch_loss = train_epoch_loss / (len(train_loader))
                train_epoch_loss = train_epoch_loss.item()
                train_epoch_iou = train_epoch_iou / (len(train_loader))
                train_epoch_iou = train_epoch_iou.item()

                if use_scheduler:
                    if scheduler_name == "ReduceLROnPlateau":
                        scheduler.step(train_epoch_loss)
                    else:
                        scheduler.step()

                sly.logger.info(f"Epoch {epoch} train loss: {train_epoch_loss}")
                sly.logger.info(f"Epoch {epoch} train IoU: {train_epoch_iou}")

                train_loss_history.append(train_epoch_loss)
                train_iou_history.append(train_epoch_iou)

                # add train loss and iou values to charts
                if train_metric_charts_f.is_hidden():
                    train_metric_charts_f.show()
                train_metric_charts.add_scalar(
                    "train loss/train loss", round(train_epoch_loss, 2), epoch
                )
                train_metric_charts.add_scalar(
                    "train IoU/train IoU", round(train_epoch_iou, 2), epoch
                )

            if (epoch % val_freq == 0 or epoch == n_epochs) and not g.stop_training:
                sam2_model.eval()
                val_epoch_loss = 0
                val_epoch_iou = 0
                with progress_bar_batches(
                    message="Validation batches:", total=n_val_batches
                ) as batch_pbar:
                    for batch_number, batch in enumerate(val_loader):
                        if g.stop_training:
                            break
                        n_points = random.randint(min_points, max_points)
                        images, masks = batch
                        batch_images, batch_masks, batch_points, batch_labels = (
                            generate_prompts_for_batch(masks, images, n_points)
                        )

                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                predictor.set_image_batch(batch_images)
                                prd_masks, _, _ = predictor.predict_batch(
                                    point_coords_batch=batch_points,
                                    point_labels_batch=batch_labels,
                                    multimask_output=True,
                                    return_logits=True,
                                )
                            prd_masks = torch.tensor(np.array(prd_masks))
                            prd_mask = torch.sigmoid(prd_masks[:, 0])
                            gt_mask = torch.tensor(
                                np.array(batch_masks).astype(np.float32)
                            )
                            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                            iou = inter / (
                                gt_mask.sum(1).sum(1)
                                + (prd_mask > 0.5).sum(1).sum(1)
                                - inter
                            )
                            iou = np.mean(iou.cpu().detach().numpy())

                            if loss_name != "CrossEntropyLoss":
                                gt_mask = gt_mask.type(torch.int64)

                            seg_loss = loss_func(prd_mask, gt_mask)
                            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                            loss = seg_loss + score_loss * 0.05
                            val_epoch_loss += loss
                            val_epoch_iou += iou

                        batch_pbar.update()
                        if (batch_number + 1) % n_train_batches == 0:
                            batch_pbar.reset()

                if not g.stop_training:

                    val_epoch_loss = val_epoch_loss / (len(val_loader))
                    val_epoch_loss = val_epoch_loss.item()
                    val_epoch_iou = val_epoch_iou / (len(val_loader))
                    val_epoch_iou = val_epoch_iou.item()

                    val_loss_history.append(val_epoch_loss)
                    val_iou_history.append(val_epoch_iou)

                    sly.logger.info(f"Epoch {epoch} val loss: {val_epoch_loss}")
                    sly.logger.info(f"Epoch {epoch} val IoU: {val_epoch_iou}")

                    # add val loss and iou values to charts
                    if val_metric_charts.is_hidden():
                        val_metric_charts.show()
                    val_metric_charts.add_scalar(
                        "validation loss/val loss", round(val_epoch_loss, 2), epoch
                    )
                    val_metric_charts.add_scalar(
                        "validation IoU/val IoU", round(val_epoch_iou, 2), epoch
                    )

                    sam2_model.sam_mask_decoder.train(True)
                    sam2_model.sam_prompt_encoder.train(True)
                    sam2_model.image_encoder.train(False)

                    # save checkpoint (if best)
                    if val_epoch_iou > best_val_iou:
                        model_dict = {
                            "model": predictor.model.state_dict(),
                            "config": config_path,
                        }
                        torch.save(model_dict, checkpoint_save_path)
                        best_val_iou = val_epoch_iou
                        best_val_epoch = epoch

                    if val_epoch_iou > prev_val_epoch_iou:
                        no_improvements = 0
                    else:
                        no_improvements += 1
                        if no_improvements == patience:
                            patience_warning_text = " ".join(
                                (
                                    f"There were no observable improvements for {no_improvements} validation epochs.",
                                    "The patience limit has been reached, the training process will be stopped early.",
                                )
                            )
                            sly.logger.info(patience_warning_text)
                            early_stopping_warning.set(
                                text=patience_warning_text, status="warning"
                            )
                            early_stopping_warning.show()
                            g.stop_training = True

                    prev_val_epoch_iou = val_epoch_iou

            epoch_pbar.update()

    sly.logger.info("Successfully finished training process")

    # generate image predictions
    if os.path.exists(checkpoint_save_path):
        finetuned_sam2_model = build_sam2(
            config_path, checkpoint_save_path, device=device
        )
        finetuned_predictor = SAM2ImagePredictor(finetuned_sam2_model)
        predictions_paths = generate_predictions(
            val_set,
            finetuned_predictor,
            project_meta,
            min_points,
            max_points,
            progress_bar_predictions,
        )

        for i, image_pair in enumerate(predictions_paths):
            local_gt_path, local_pred_path = image_pair
            remote_gt_path = os.path.join(
                sly.output.RECOMMENDED_EXPORT_PATH,
                sly.app.fastapi.get_name_from_env(),
                str(g.app_session_id),
                os.path.basename(local_gt_path),
            )
            gt_info = api.file.upload(team_id, local_gt_path, remote_gt_path)
            remote_pred_path = os.path.join(
                sly.output.RECOMMENDED_EXPORT_PATH,
                sly.app.fastapi.get_name_from_env(),
                str(g.app_session_id),
                os.path.basename(local_pred_path),
            )
            pred_info = api.file.upload(team_id, local_pred_path, remote_pred_path)
            if sly.is_production():
                gt_id = predictions_gallery.append(
                    image_url=gt_info.storage_path,
                    title="ground truth",
                )
                pred_id = predictions_gallery.append(
                    image_url=pred_info.storage_path,
                    title="predicted",
                )
            else:
                gt_id = predictions_gallery.append(
                    image_url=gt_info.full_storage_url,
                    title="ground truth",
                )
                pred_id = predictions_gallery.append(
                    image_url=pred_info.full_storage_url,
                    title="predicted",
                )
            predictions_gallery.sync_images([gt_id, pred_id])
            if i == 0:
                predictions_gallery_f.show()
    else:
        sly.logger.warn(
            "Model checkpoint not found, unable to generate model predictions"
        )

    # generate .csv file with training history
    if len(train_loss_history) > 0:
        train_history_df = generate_history_df(
            train_loss_history, train_iou_history, mode="train"
        )
        train_history_df_path = os.path.join(g.train_artifacts_dir, "train_history.csv")
        train_history_df.to_csv(train_history_df_path)
    if len(val_loss_history) > 0:
        val_history_df = generate_history_df(
            val_loss_history, val_iou_history, mode="val"
        )
        val_history_df_path = os.path.join(g.train_artifacts_dir, "val_history.csv")
        val_history_df.to_csv(val_history_df_path)

    # save link to app ui
    app_url = f"/apps/sessions/{g.app_session_id}"
    app_link_path = os.path.join(g.train_artifacts_dir, "open_app.lnk")
    with open(app_link_path, "w") as text_file:
        print(app_url, file=text_file)

    # upload training artifacts to team files
    upload_artifacts_dir = os.path.join(
        "SAM2",
        project_info.name,
        str(g.app_session_id),
    )

    def upload_monitor(monitor, api: sly.Api, progress: sly.Progress):
        value = monitor.bytes_read
        if progress.total == 0:
            progress.set(value, monitor.len, report=False)
        else:
            progress.set_current_value(value, report=False)
        artifacts_pbar.update(progress.current - artifacts_pbar.n)

    local_files = sly.fs.list_files_recursively(g.train_artifacts_dir)
    total_size = sum([sly.fs.get_file_size(file_path) for file_path in local_files])
    progress = sly.Progress(
        message="",
        total_cnt=total_size,
        is_size=True,
    )
    progress_cb = partial(upload_monitor, api=api, progress=progress)
    with progress_bar_upload_artifacts(
        message="Uploading train artifacts to Team Files...",
        total=total_size,
        unit="bytes",
        unit_scale=True,
    ) as artifacts_pbar:
        remote_artifacts_dir = api.file.upload_directory(
            team_id=sly.env.team_id(),
            local_dir=g.train_artifacts_dir,
            remote_dir=upload_artifacts_dir,
            progress_size_cb=progress_cb,
        )
    progress_bar_upload_artifacts.hide()
    file_info = api.file.get_info_by_path(
        sly.env.team_id(), remote_artifacts_dir + "/open_app.lnk"
    )
    train_artifacts_folder.set(file_info)

    progress_bar_epochs.hide()
    progress_bar_batches.hide()
    start_training_button.loading = False
    start_training_button.disable()
    train_done.show()
    logs_button.disable()
    card_train_artifacts.unlock()
    card_train_artifacts.uncollapse()

    if best_val_iou > 0:
        sly.logger.info(
            f"Best validation IoU: {best_val_iou} (reached at epoch {best_val_epoch})"
        )

    if g.stop_training:
        stop_training_button.loading = False
    stop_training_button.disable()

    # delete app data since it is no longer needed
    if sly.is_production():
        sly.fs.remove_dir(g.app_data_dir)

    # set task output
    sly.output.set_directory(remote_artifacts_dir)
    # stop app
    app.stop()
