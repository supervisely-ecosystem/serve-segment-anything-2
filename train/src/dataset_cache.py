import asyncio
import os
from typing import List
import supervisely as sly
from supervisely.app.widgets import Progress
import src.globals as g
from supervisely.project.download import (
    download_to_cache,
    copy_from_cache,
    get_cache_size,
)
from supervisely.project.project import Dataset


def _no_cache_download(
    api: sly.Api,
    project_info: sly.ProjectInfo,
    dataset_infos: List[sly.DatasetInfo],
    progress: Progress,
    semaphore: asyncio.Semaphore = None,
):
    dataset_ids = [dataset_info.id for dataset_info in dataset_infos]
    total = sum([dataset_info.images_count for dataset_info in dataset_infos])
    try:
        with progress(message="Downloading input data...", total=total) as pbar:
            sly.download_async(
                api=api,
                project_id=project_info.id,
                dest_dir=g.project_dir,
                semaphore=semaphore,
                dataset_ids=dataset_ids,
                progress_cb=pbar.update,
                save_images=True,
                save_image_info=True,
            )
    except Exception:
        api.logger.warning(
            "Failed to download project using async download. Trying sync download..."
        )
        with progress(message="Downloading input data...", total=total) as pbar:
            sly.download_project(
                api=api,
                project_id=project_info.id,
                dest_dir=g.project_dir,
                dataset_ids=dataset_ids,
                log_progress=True,
                progress_cb=pbar.update,
                save_images=True,
                save_image_info=True,
                resume_download=True,
            )


def _get_dataset_parents(api, dataset_infos, dataset_id):
    dataset_infos_dict = {info.id: info for info in dataset_infos}
    this_dataset_info = dataset_infos_dict.get(
        dataset_id, api.dataset.get_info_by_id(dataset_id)
    )
    if this_dataset_info.parent_id is None:
        return []
    parent = _get_dataset_parents(
        api, list(dataset_infos_dict.values()), this_dataset_info.parent_id
    )
    this_parent_name = dataset_infos_dict.get(
        this_dataset_info.parent_id, api.dataset.get_info_by_id(dataset_id)
    ).name
    return [*parent, this_parent_name]


def _get_dataset_path(api, dataset_infos, dataset_id):
    parents = _get_dataset_parents(api, dataset_infos, dataset_id)
    dataset_infos_dict = {info.id: info for info in dataset_infos}
    this_dataset_info = dataset_infos_dict.get(
        dataset_id, api.dataset.get_info_by_id(dataset_id)
    )
    return Dataset._get_dataset_path(this_dataset_info.name, parents)


def download_project(
    api: sly.Api,
    project_info: sly.ProjectInfo,
    dataset_infos: List[sly.DatasetInfo],
    use_cache: bool,
    progress: Progress,
):
    if os.path.exists(g.project_dir):
        sly.fs.clean_dir(g.project_dir)
    if not use_cache:
        return _no_cache_download(api, project_info, dataset_infos, progress)
    try:
        # get images count
        total = sum([ds_info.images_count for ds_info in dataset_infos])
        # download
        with progress(message="Downloading input data...", total=total) as pbar:
            download_to_cache(
                api=api,
                project_id=project_info.id,
                dataset_infos=dataset_infos,
                log_progress=True,
                progress_cb=pbar.update,
            )
        # copy datasets from cache
        total = sum(
            [
                get_cache_size(
                    project_info.id, _get_dataset_path(api, dataset_infos, ds.id)
                )
                for ds in dataset_infos
            ]
        )
        with progress(
            message="Retreiving data from cache...",
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            dataset_names = [
                _get_dataset_path(api, dataset_infos, ds_info.id)
                for ds_info in dataset_infos
            ]
            copy_from_cache(
                project_id=project_info.id,
                dest_dir=g.project_dir,
                dataset_names=dataset_names,
                progress_cb=pbar.update,
            )
    except Exception:
        sly.logger.warning(
            f"Failed to retreive project from cache. Downloading it...", exc_info=True
        )
        if os.path.exists(g.project_dir):
            sly.fs.clean_dir(g.project_dir)
        _no_cache_download(api, project_info, dataset_infos, progress)
