import functools
import json
import os
import threading
import time
import traceback
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Literal
import uuid

import cv2
import mock
import numpy as np
import supervisely as sly
import torch
from cacheout import Cache
from cachetools import LRUCache
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Request, Response, status
from fastapi.responses import StreamingResponse
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from supervisely._utils import is_debug_with_sly_net, rand_str
from supervisely.app.content import get_data_dir
from supervisely.app.widgets import Field, Switch
from supervisely.imaging import image as sly_image
from supervisely.imaging.color import generate_rgb
from supervisely.io.fs import mkdir, remove_dir, silent_remove
from supervisely.nn.inference.interactive_segmentation import functional
from supervisely.sly_logger import logger
from supervisely.app.widgets import SelectString, Field
from supervisely.api.module_api import ApiField
from supervisely.nn.inference.inference import (
    _convert_sly_progress_to_dict,
    _get_log_extra_for_inference_request,
)
from supervisely.api.video_annotation_tool_api import VideoAnnotationToolAction


load_dotenv("supervisely.env")
load_dotenv("debug.env")
api = sly.Api()
root_source_path = str(Path(__file__).parents[1])
debug_session = bool(os.environ.get("DEBUG_SESSION", False))
model_data_path = os.path.join(root_source_path, "models", "models.json")
UPLOAD_SLEEP_TIME = 0.1
NOTIFY_SLEEP_TIME = 0.1


def notqdm(iterable, *args, **kwargs):
    """
    replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests
    """
    return iterable


class SegmentAnything2(sly.nn.inference.PromptableSegmentation):
    def add_content_to_custom_tab(self, gui):
        self.select_config = SelectString(
            values=[
                "sam2.1_hiera_t.yaml",
                "sam2.1_hiera_s.yaml",
                "sam2.1_hiera_b+.yaml",
                "sam2.1_hiera_l.yaml",
            ]
        )
        select_config_f = Field(self.select_config, "Select SAM 2 model config")
        return select_config_f

    def add_content_to_pretrained_tab(self, gui):
        self.use_bbox = Switch(switched=True)
        use_bbox_field = Field(
            content=self.use_bbox,
            title="Use bounding box prompt",
            description=(
                "Define whether to use bounding box prompt when labeling images and videos or not. "
                "If turned off, then only point prompts (positive and negative clicks) will be used. "
                "Adding bounding box prompt can be useful when labeling entire objects, while using "
                "only point prompts can be better when segmenting specific parts of objects."
            ),
        )
        return use_bbox_field

    def support_custom_models(self):
        return True

    def get_models(self, mode="table"):
        model_data = sly.json.load_json_file(model_data_path)
        if mode == "table":
            for element in model_data:
                del element["weights_path"]
                del element["config"]
            return model_data
        elif mode == "info":
            models_data_processed = {}
            for element in model_data:
                models_data_processed[element["Model"]] = {
                    "weights_path": element["weights_path"],
                    "config": element["config"],
                }
            return models_data_processed

    def get_weights_path_and_config(self):
        models_data = self.get_models(mode="info")
        selected_model = self.gui.get_checkpoint_info()["Model"]
        weights_path = models_data[selected_model]["weights_path"]
        if debug_session:
            weights_path = "." + weights_path
        config = models_data[selected_model]["config"]
        return weights_path, config

    def get_models_table_row_idx_and_config(self, weights_path):
        if weights_path.endswith("tiny.pt"):
            idx = 0
            config = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif weights_path.endswith("small.pt"):
            idx = 1
            config = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif weights_path.endswith("base_plus.pt"):
            idx = 2
            config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif weights_path.endswith("large.pt"):
            idx = 3
            config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        return idx, config

    def load_on_device(
        self,
        model_dir: str = "app_data",
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
        from_api=False,
        model_source=None,
        weights_path=None,
        config=None,
        custom_link=None,
    ):
        if not from_api:
            model_source = self.gui.get_model_source()
        if model_source == "Pretrained models":
            # get weights path and config
            if from_api:
                self.weights_path = weights_path
                row_idx, self.config = self.get_models_table_row_idx_and_config(
                    weights_path
                )
                self.gui._models_table.select_row(row_idx)
            else:
                self.weights_path, self.config = self.get_weights_path_and_config()
                if sly.is_development():
                    self.weights_path = "." + self.weights_path
        elif model_source == "Custom models":
            if not from_api:
                custom_link = self.gui.get_custom_link()
            else:
                self.gui._tabs.set_active_tab("Custom models")
                self.gui._model_path_input.set_value(custom_link)
                file_info = api.file.get_info_by_path(sly.env.team_id(), custom_link)
                self.gui._file_thumbnail.set(file_info)
            weights_file_name = os.path.basename(custom_link)
            self.weights_path = os.path.join(model_dir, weights_file_name)
            if not sly.fs.file_exists(self.weights_path):
                self.download(
                    src_path=custom_link,
                    dst_path=self.weights_path,
                )
            if from_api:
                self.config = config
                self.select_config.set_value(config.split("/")[2])
            else:
                self.config = self.select_config.get_value()
                self.config = "configs/sam2.1/" + self.config
        # build model
        self.sam = build_sam2(self.config, self.weights_path, device=device)
        # load model on device
        if device != "cpu":
            if device == "cuda":
                torch.cuda.set_device(0)
            else:
                torch.cuda.set_device(int(device[-1]))
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            self.torch_device = torch.device(device)
            self.sam.to(device=self.torch_device)
        else:
            self.sam.to(device=device)
        # build predictor
        self.predictor = SAM2ImagePredictor(self.sam)
        self.video_predictor = None
        # define class names
        self.class_names = ["object_mask"]
        # list for storing mask colors
        self.mask_colors = [[255, 0, 0]]
        # variable for storing image ids from previous inference iterations
        self.previous_image_id = None
        # dict for storing model variables to avoid unnecessary calculations
        self.model_cache = Cache(maxsize=100, ttl=5 * 60)
        # set variables for smart tool mode
        self._inference_image_lock = threading.Lock()

        # TODO: add maxsize after discuss
        self._inference_image_cache = Cache(ttl=60)
        self._init_mask_cache = LRUCache(maxsize=100)  # cache of sly.Bitmaps

    def get_info(self):
        info = super().get_info()
        info["videos_support"] = True
        info["async_video_inference_support"] = True
        return info

    def get_classes(self) -> List[str]:
        return self.class_names

    @property
    def model_meta(self):
        if self._model_meta is None:
            self._model_meta = sly.ProjectMeta(
                [sly.ObjClass(self.class_names[0], sly.Bitmap, [255, 0, 0])]
            )
            self._get_confidence_tag_meta()
        return self._model_meta

    def set_image_data(self, input_image, settings):
        if settings["input_image_id"] != self.previous_image_id:
            if settings["input_image_id"] not in self.model_cache:
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    self.predictor.set_image(input_image)
                cache_features = self.predictor._features.copy()
                cache_features["image_embed"] = (
                    cache_features["image_embed"].detach().cpu()
                )
                cache_features["high_res_feats"] = [
                    element.detach().cpu()
                    for element in cache_features["high_res_feats"]
                ]
                self.model_cache.set(
                    settings["input_image_id"],
                    {
                        "features": cache_features,
                        "original_size": self.predictor._orig_hw,
                    },
                )
            else:
                cached_data = self.model_cache.get(settings["input_image_id"])
                cached_data["features"]["image_embed"] = cached_data["features"][
                    "image_embed"
                ].to(self.torch_device)
                cached_data["features"]["high_res_feats"] = [
                    element.to(self.torch_device)
                    for element in cached_data["features"]["high_res_feats"]
                ]
                self.predictor._features = cached_data["features"]
                self.predictor._orig_hw = cached_data["original_size"]

    def _deserialize_geometry(self, data: dict):
        geometry_type_str = data["type"]
        geometry_json = data["data"]
        return sly.deserialize_geometry(geometry_type_str, geometry_json)

    def set_cuda_properties(self):
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionMask]:
        self.set_cuda_properties()
        # prepare input data
        input_image = sly.image.read(image_path)
        # list for storing preprocessed masks
        predictions = []
        if self._model_meta is None:
            self._model_meta = self.model_meta
        if settings["mode"] == "raw":
            # build mask generator and generate masks
            mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam,
                points_per_side=settings["points_per_side"],
                points_per_batch=settings["points_per_batch"],
                pred_iou_thresh=settings["pred_iou_thresh"],
                stability_score_thresh=settings["stability_score_thresh"],
                stability_score_offset=settings["stability_score_offset"],
                box_nms_thresh=settings["box_nms_thresh"],
                crop_n_layers=settings["crop_n_layers"],
                crop_nms_thresh=settings["crop_nms_thresh"],
                crop_overlap_ratio=settings["crop_overlap_ratio"],
                crop_n_points_downscale_factor=settings[
                    "crop_n_points_downscale_factor"
                ],
                min_mask_region_area=settings["min_mask_region_area"],
                output_mode=settings["output_mode"],
                use_m2m=settings["use_m2m"],
            )
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks = mask_generator.generate(input_image)
            for i, mask in enumerate(masks):
                # get predicted mask
                mask = mask["segmentation"]
                predictions.append(
                    sly.nn.PredictionMask(class_name="object_mask", mask=mask)
                )
        elif settings["mode"] == "bbox":
            # get bbox coordinates
            if "rectangle" not in settings:
                bbox_coordinates = settings["bbox_coordinates"]
            else:
                rectangle = sly.Rectangle.from_json(settings["rectangle"])
                bbox_coordinates = [
                    rectangle.top,
                    rectangle.left,
                    rectangle.bottom,
                    rectangle.right,
                ]
            # transform bbox from yxyx to xyxy format
            bbox_coordinates = [
                bbox_coordinates[1],
                bbox_coordinates[0],
                bbox_coordinates[3],
                bbox_coordinates[2],
            ]
            bbox_coordinates = np.array(bbox_coordinates)
            # get bbox class name and add new class to model meta if necessary
            class_name = settings["bbox_class_name"] + "_mask"
            if not self._model_meta.get_obj_class(class_name):
                self.class_names.append(class_name)
                new_class = sly.ObjClass(class_name, sly.Bitmap, [255, 0, 0])
                self._model_meta = self._model_meta.add_obj_class(new_class)
            # generate image embedding - model will remember this embedding and use it for subsequent mask prediction
            self.set_image_data(input_image, settings)
            self.previous_image_id = settings["input_image_id"]
            # get predicted mask
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks, _, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=bbox_coordinates[None, :],
                    multimask_output=False,
                )
            mask = masks[0]
            predictions.append(sly.nn.PredictionMask(class_name=class_name, mask=mask))
        elif settings["mode"] == "points":
            # get point coordinates
            point_coordinates = settings["point_coordinates"]
            point_coordinates = np.array(point_coordinates)
            # get point labels
            point_labels = settings["point_labels"]
            point_labels = np.array(point_labels)
            # set class name
            if settings["points_class_name"] not in [None, "None"]:
                class_name = settings["points_class_name"]
            else:
                class_name = self.class_names[0]
            # add new class to model meta if necessary
            if not self._model_meta.get_obj_class(class_name):
                color = generate_rgb(self.mask_colors)
                self.mask_colors.append(color)
                self.class_names.append(class_name)
                new_class = sly.ObjClass(class_name, sly.Bitmap, color)
                self._model_meta = self._model_meta.add_obj_class(new_class)
            # generate image embedding - model will remember this embedding and use it for subsequent mask prediction
            self.set_image_data(input_image, settings)
            self.previous_image_id = settings["input_image_id"]
            # get predicted masks
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                if len(point_labels) > 1:
                    masks, _, _ = self.predictor.predict(
                        point_coords=point_coordinates,
                        point_labels=point_labels,
                        multimask_output=False,
                    )
                    mask = masks[0]
                else:
                    masks, scores, logits = self.predictor.predict(
                        point_coords=point_coordinates,
                        point_labels=point_labels,
                        multimask_output=True,
                    )
                    max_score_ind = np.argmax(scores)
                    mask = masks[max_score_ind]
            predictions.append(sly.nn.PredictionMask(class_name=class_name, mask=mask))
        elif settings["mode"] == "combined":
            # get point coordinates
            point_coordinates = settings["point_coordinates"]
            point_coordinates = np.array(point_coordinates)
            # get point labels
            point_labels = settings["point_labels"]
            point_labels = np.array(point_labels)
            # get bbox coordinates
            bbox_coordinates = settings["bbox_coordinates"]
            # transform bbox from yxyx to xyxy format
            bbox_coordinates = [
                bbox_coordinates[1],
                bbox_coordinates[0],
                bbox_coordinates[3],
                bbox_coordinates[2],
            ]
            bbox_coordinates = np.array(bbox_coordinates)
            # get bbox class name and add new class to model meta if necessary
            class_name = settings["bbox_class_name"] + "_mask"
            if not self._model_meta.get_obj_class(class_name):
                self.class_names.append(class_name)
                new_class = sly.ObjClass(class_name, sly.Bitmap, [255, 0, 0])
                self._model_meta = self._model_meta.add_obj_class(new_class)
            # generate image embedding - model will remember this embedding and use it for subsequent mask prediction
            self.set_image_data(input_image, settings)
            init_mask = settings.get("init_mask")
            # get predicted masks
            if (
                settings["input_image_id"] in self.model_cache
                and (
                    self.model_cache.get(settings["input_image_id"]).get(
                        "previous_bbox"
                    )
                    == bbox_coordinates
                ).all()
                and self.previous_image_id == settings["input_image_id"]
            ):
                # get mask from previous predicton and use at as an input for new prediction
                mask_input = self.model_cache.get(settings["input_image_id"])[
                    "mask_input"
                ]
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    if len(point_labels) > 1:
                        masks, scores, logits = self.predictor.predict(
                            point_coords=point_coordinates,
                            point_labels=point_labels,
                            box=bbox_coordinates[None, :],
                            mask_input=mask_input[None, :, :],
                            multimask_output=False,
                        )
                    else:
                        masks, scores, logits = self.predictor.predict(
                            point_coords=point_coordinates,
                            point_labels=point_labels,
                            box=bbox_coordinates[None, :],
                            mask_input=mask_input[None, :, :],
                            multimask_output=True,
                        )
                        max_score_ind = np.argmax(scores)
                        masks = [masks[max_score_ind]]
            elif init_mask is not None:

                mask_input = torch.tensor(init_mask).float()

                # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
                out_scale, out_bias = 10.0, -10.0  # sigmoid(-10.0)=4.5398e-05
                high_res_masks = mask_input * out_scale + out_bias
                mask_input = torch.nn.functional.interpolate(
                    high_res_masks.expand(
                        (1, 1, *high_res_masks.shape)
                    ),  # Change from HxW to BxCxHxW
                    size=(
                        self.predictor.model.image_size // 4,
                        self.predictor.model.image_size // 4,
                    ),
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                ).squeeze((0))

                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    if len(point_labels) > 1:
                        masks, scores, logits = self.predictor.predict(
                            point_coords=point_coordinates,
                            point_labels=point_labels,
                            box=bbox_coordinates[None, :],
                            mask_input=mask_input[None, :, :],
                            multimask_output=False,
                        )
                    else:
                        masks, scores, logits = self.predictor.predict(
                            point_coords=point_coordinates,
                            point_labels=point_labels,
                            box=bbox_coordinates[None, :],
                            mask_input=mask_input[None, :, :],
                            multimask_output=True,
                        )
                        max_score_ind = np.argmax(scores)
                        masks = [masks[max_score_ind]]
            else:
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    if len(point_labels) > 1:
                        masks, scores, logits = self.predictor.predict(
                            point_coords=point_coordinates,
                            point_labels=point_labels,
                            box=bbox_coordinates[None, :],
                            multimask_output=False,
                        )
                    else:
                        masks, scores, logits = self.predictor.predict(
                            point_coords=point_coordinates,
                            point_labels=point_labels,
                            box=bbox_coordinates[None, :],
                            multimask_output=True,
                        )
                        max_score_ind = np.argmax(scores)
                        masks = [masks[max_score_ind]]
            # save bbox ccordinates and mask to cache
            if settings["input_image_id"] in self.model_cache:
                image_id = settings["input_image_id"]
                cached_data = self.model_cache.get(image_id)
                cached_data["previous_bbox"] = bbox_coordinates
                if len(point_labels) > 1:
                    cached_data["mask_input"] = logits[0]
                else:
                    cached_data["mask_input"] = logits[max_score_ind]
                self.model_cache.set(image_id, cached_data)
            # update previous_image_id variable
            self.previous_image_id = settings["input_image_id"]
            mask = masks[0]
            predictions.append(sly.nn.PredictionMask(class_name=class_name, mask=mask))
        return predictions

    def get_bitmap_center(self, bitmap):
        contours, _ = cv2.findContours(
            bitmap.data.astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        moments = cv2.moments(contours[0])
        cx = int((moments["m10"] + 1e-7) / (moments["m00"] + 1e-7))
        cy = int((moments["m01"] + 1e-7) / (moments["m00"] + 1e-7))
        return [cx, cy]

    def generate_artificial_prompt(self, bitmap, frame_np):
        # generate point prompts
        mask = bitmap.data
        origin_row, origin_col = bitmap.origin.row, bitmap.origin.col
        row_indexes, col_indexes = np.where(mask)
        point_coordinates, point_labels = [], []
        for i in range(3):
            idx = np.random.randint(0, len(row_indexes))
            row_index, col_index = row_indexes[idx], col_indexes[idx]
            row_index += origin_row
            col_index += origin_col
            point_coordinates.append([col_index, row_index])
            point_labels.append(1)
        prompt = {
            "point_coordinates": point_coordinates,
            "point_labels": point_labels,
        }
        if self.use_bbox.is_switched():
            # generate box prompt
            rectangle = bitmap.to_bbox()
            padding = 0.03
            # extract original size
            original_w, original_h = rectangle.width, rectangle.height
            center = (rectangle.center.col, rectangle.center.row)
            # apply padding
            padded_w, padded_h = (1 + padding) * original_w, (1 + padding) * original_h
            padded_left = center[0] - (round(padded_w / 2))
            padded_top = center[1] - (round(padded_h / 2))
            padded_right = center[0] + (round(padded_w / 2))
            padded_bottom = center[1] + (round(padded_h / 2))
            # check if padded bbox is not out of image bounds
            image_h, image_w = frame_np.shape[0], frame_np.shape[1]
            padded_left = max(1, padded_left)
            padded_top = max(1, padded_top)
            padded_right = min(image_w - 1, padded_right)
            padded_bottom = min(image_h - 1, padded_bottom)
            bbox = [
                padded_left,
                padded_top,
                padded_right,
                padded_bottom,
            ]
            prompt["bbox"] = bbox
        return prompt

    def get_smarttool_input(self, figure: sly.FigureInfo):
        if figure.meta is None:
            return None
        smarttool_input = figure.meta.get("smartToolInput", None)
        if smarttool_input is None:
            return None
        crop = smarttool_input["crop"]
        crop = [*crop[0], *crop[1]]
        positive = smarttool_input["positive"]
        negative = smarttool_input["negative"]
        visible = smarttool_input["visible"]
        return crop, positive, negative, visible

    @mock.patch("sam2.sam2_video_predictor.tqdm", notqdm)
    @mock.patch("sam2.utils.misc.tqdm", notqdm)
    def _track_api(self, api: sly.Api, context: dict):
        self.set_cuda_properties()
        # TODO: Add clicks support
        video_id = context["videoId"]
        start_frame = context["frameIndex"]
        n_frames = context["frames"]
        input_geometries = context["input_geometries"]
        direction = 1 if context.get("direction", "forward") == "forward" else -1
        log_extra = {
            "video_id": video_id,
            "start_frame": start_frame,
            "frames": n_frames,
            "direction": direction,
        }
        sly.logger.info("Starting tracking process...", extra=log_extra)
        end_frame = start_frame + n_frames * direction
        frames_indexes = list(range(start_frame, end_frame + direction, direction))

        # start background task for caching frames
        api.logger.debug("Starting cache task for video %s", video_id, extra=log_extra)
        if self.cache.is_persistent:
            # if cache is persistent, run cache task for whole video
            frame_range = None
        else:
            # if cache is not persistent, run cache task for range of frames
            frame_range = [start_frame, end_frame]
            if direction == -1:
                frame_range = frame_range[::-1]
        self.cache.run_cache_task_manually(
            api,
            frame_range,
            video_id=video_id,
        )

        temp_frames_dir = f"frames/{rand_str(10)}"
        # save frames to directory
        api.logger.debug("Saving frames to directory...", extra=log_extra)
        mkdir(temp_frames_dir, remove_content_if_exists=True)
        self.cache.download_frames_to_paths(
            api,
            video_id,
            frames_indexes,
            [f"{temp_frames_dir}/{i}.jpg" for i in range(n_frames + 1)],
        )

        # initialize model
        if not self.video_predictor:
            self.video_predictor = build_sam2_video_predictor(
                self.config, self.weights_path
            )

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.video_predictor.init_state(
                video_path=temp_frames_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=True,
            )

        for i, input_geom_data in enumerate(input_geometries):
            geometry = self._deserialize_geometry(input_geom_data)
            if not isinstance(geometry, sly.Bitmap) and not isinstance(
                geometry, sly.Polygon
            ):
                raise TypeError(
                    f"This app does not support {geometry.geometry_name()} tracking"
                )
            # convert polygon to bitmap
            if isinstance(geometry, sly.Polygon):
                polygon_obj_class = sly.ObjClass("polygon", sly.Polygon)
                polygon_label = sly.Label(geometry, polygon_obj_class)
                bitmap_obj_class = sly.ObjClass("bitmap", sly.Bitmap)
                bitmap_label = polygon_label.convert(bitmap_obj_class)[0]
                geometry = bitmap_label.geometry

            first_frame = sly_image.read(f"{temp_frames_dir}/0.jpg")
            prompt = self.generate_artificial_prompt(geometry, first_frame)
            smarttool_input = (prompt["bbox"], prompt["point_coordinates"], [], True)

            # bbox - ltrb
            # points - col, row
            bbox, positive_clicks, negative_clicks, _ = smarttool_input
            if not self.use_bbox.is_switched():
                bbox = None
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=i,
                    points=positive_clicks + negative_clicks,
                    labels=[1] * len(positive_clicks) + [0] * len(negative_clicks),
                    box=bbox,
                )

        results = []
        # run propagation throughout the video
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for (
                out_frame_idx,
                _,
                out_mask_logits,
            ) in self.video_predictor.propagate_in_video(inference_state):
                # skip first frame prediction
                if out_frame_idx == 0:
                    continue
                results.append([])
                for masks in out_mask_logits:
                    masks = (masks > 0.0).cpu().numpy()
                    sum_mask = np.any(masks, axis=0)
                    geometry = sly.Bitmap(sum_mask, extra_validation=False)
                    results[-1].append(
                        {"type": geometry.geometry_name(), "data": geometry.to_json()}
                    )

        self.video_predictor.reset_state(inference_state)
        return results

    def _track(
        self,
        api: sly.Api,
        context: Dict,
    ):
        self.set_cuda_properties()
        video_id = context["videoId"]
        track_id = context["trackId"]
        n_frames = context["frames"]
        start_frame = context["frameIndex"]
        figure_ids = context["figureIds"]
        direction = 1 if context.get("direction", "forward") == "forward" else -1
        log_extra = {
            "video_id": video_id,
            "track_id": track_id,
            "start_frame": start_frame,
            "frames": n_frames,
            "figure_ids": figure_ids,
            "direction": direction,
        }
        sly.logger.info("Starting tracking process...", extra=log_extra)
        end_frame = start_frame + n_frames * direction
        frames_indexes = list(range(start_frame, end_frame + direction, direction))
        progress = sly.Progress(
            "Tracking progress", total_cnt=n_frames + 1 + n_frames * len(figure_ids)
        )

        # start background task for caching frames
        api.logger.debug("Starting cache task for video %s", video_id, extra=log_extra)
        if self.cache.is_persistent:
            # if cache is persistent, run cache task for whole video
            frame_range = None
        else:
            # if cache is not persistent, run cache task for range of frames
            frame_range = [start_frame, end_frame]
            if direction == -1:
                frame_range = frame_range[::-1]
        self.cache.run_cache_task_manually(
            api,
            frame_range,
            video_id=video_id,
        )

        # load figures
        api.logger.debug("Loading figures...", extra=log_extra)
        video_info = api.video.get_info_by_id(video_id)
        figures = api.video.figure.get_by_ids(video_info.dataset_id, figure_ids)
        figure_id_to_object_id = {figure.id: figure.object_id for figure in figures}

        notify_stop = threading.Event()

        def _notify_loop():
            _start_frame = start_frame if direction == 1 else end_frame
            _end_frame = end_frame if direction == 1 else start_frame
            last_notify = 0
            while not notify_stop.is_set():
                if progress.current > last_notify:
                    api.video.notify_progress(
                        track_id,
                        video_id,
                        _start_frame,
                        _end_frame,
                        progress.current,
                        progress.total,
                    )
                    last_notify = progress.current
                time.sleep(NOTIFY_SLEEP_TIME)
            if progress.current > last_notify:
                api.video.notify_progress(
                    track_id,
                    video_id,
                    _start_frame,
                    _end_frame,
                    progress.current,
                    progress.total,
                )

        notify_thread = threading.Thread(target=_notify_loop, daemon=True)
        notify_thread.start()
        inference_state = None
        upload_thread = None
        temp_frames_dir = f"frames/{track_id}"
        save_frames_current = 0

        def _progress_cb(cnt=1):
            nonlocal save_frames_current
            for _ in range(cnt):
                save_frames_current += 1
                api.logger.debug(
                    "Saving frames to directory: %d/%d",
                    save_frames_current,
                    n_frames + 1,
                    extra={**log_extra},
                )
                progress.iter_done()

        try:
            # save frames to directory
            api.logger.debug("Saving frames to directory...", extra=log_extra)
            mkdir(temp_frames_dir, remove_content_if_exists=True)
            self.cache.download_frames_to_paths(
                api,
                video_id,
                frames_indexes,
                [f"{temp_frames_dir}/{i}.jpg" for i in range(n_frames + 1)],
                progress_cb=_progress_cb,
            )

            # initialize model
            if not self.video_predictor:
                self.video_predictor = build_sam2_video_predictor(
                    self.config, self.weights_path
                )
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = self.video_predictor.init_state(
                    video_path=temp_frames_dir,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                    async_loading_frames=True,
                )

            for figure in figures:
                if figure.geometry_type != sly.Bitmap.geometry_name():
                    sly.logger.warning(
                        "Only geometries of shape mask are available for tracking",
                        extra=log_extra,
                    )
                    continue
                first_frame = sly_image.read(f"{temp_frames_dir}/0.jpg")
                geometry = sly.deserialize_geometry(
                    figure.geometry_type, figure.geometry
                )
                smarttool_input = self.get_smarttool_input(figure)
                if smarttool_input is None:
                    prompt = self.generate_artificial_prompt(geometry, first_frame)
                    smarttool_input = (
                        prompt.get("bbox"),
                        prompt["point_coordinates"],
                        [],
                        True,
                    )

                # bbox - ltrb
                # points - col, row
                bbox, positive_clicks, negative_clicks, _ = smarttool_input
                if not self.use_bbox.is_switched():
                    bbox = None
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=figure.id,
                        points=positive_clicks + negative_clicks,
                        labels=[1] * len(positive_clicks) + [0] * len(negative_clicks),
                        box=bbox,
                    )

            empty_mask_notified = False

            def _upload_single(frame_index, figure_id, mask):
                nonlocal empty_mask_notified
                mask = mask.astype(bool)
                if np.all(~mask):
                    logger.debug(
                        "Empty mask detected",
                        extra={**log_extra, "frame_index": frame_index},
                    )
                    if not empty_mask_notified:
                        try:
                            message = "The model has predicted empty mask"
                            api.video.notify_tracking_warning(
                                track_id, video_id, message
                            )
                            empty_mask_notified = True
                        except Exception as e:
                            api.logger.warning(
                                "Unable to notify about empty mask: %s",
                                str(e),
                                exc_info=True,
                                extra=log_extra,
                            )
                    return
                geometry = sly.Bitmap(mask, extra_validation=False)
                object_id = figure_id_to_object_id[figure_id]
                api.video.figure.create(
                    video_id,
                    object_id,
                    frame_index,
                    geometry.to_json(),
                    geometry.geometry_name(),
                    track_id,
                )

            upload_queue = Queue()
            upload_stop = threading.Event()
            upload_error = threading.Event()

            def _upload_loop(q: Queue, stop_event: threading.Event, upload_f: callable):
                _start_frame = start_frame if direction == 1 else end_frame
                _end_frame = end_frame if direction == 1 else start_frame
                try:
                    while True:
                        items = []
                        while not q.empty():
                            items.append(q.get_nowait())
                        if len(items) > 0:
                            for item in items:
                                upload_f(*item[:3])
                            progress.iters_done(sum(1 for item in items if item[3]))
                            continue
                        if stop_event.is_set():
                            api.video.notify_progress(
                                track_id,
                                video_id,
                                _start_frame,
                                _end_frame,
                                progress.total,
                                progress.total,
                            )
                            return
                        time.sleep(UPLOAD_SLEEP_TIME)
                except Exception as e:
                    api.logger.error(
                        "Error in upload loop: %s",
                        str(e),
                        exc_info=True,
                        extra=log_extra,
                    )
                    upload_error.set()
                    raise

            upload_thread = threading.Thread(
                target=_upload_loop,
                args=(upload_queue, upload_stop, _upload_single),
                daemon=True,
            )
            upload_thread.start()

            # run propagation throughout the video
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in self.video_predictor.propagate_in_video(inference_state):
                    # skip first frame prediction
                    if out_frame_idx == 0:
                        continue
                    frame_index = start_frame + out_frame_idx * direction
                    for figure_id, masks in zip(out_obj_ids, out_mask_logits):
                        if upload_error.is_set():
                            raise RuntimeError(
                                "Tracking is stopped due to an error in upload loop"
                            )
                        masks = (masks > 0.0).cpu().numpy()
                        for i, mask in enumerate(masks):
                            upload_queue.put((frame_index, figure_id, mask, i == 0))

        except Exception:
            raise
        else:
            sly.logger.info("Successfully finished tracking process", extra=log_extra)
        finally:
            if self.video_predictor is not None and inference_state is not None:
                # reset predictor state
                self.video_predictor.reset_state(inference_state)
            if upload_thread is not None and upload_thread.is_alive():
                upload_stop.set()
                upload_thread.join()
            if notify_thread.is_alive():
                notify_stop.set()
                notify_thread.join()
            remove_dir(temp_frames_dir)

    @mock.patch("sam2.sam2_video_predictor.tqdm", notqdm)
    @mock.patch("sam2.utils.misc.tqdm", notqdm)
    def _track_async(self, api: sly.Api, context: dict, request_uuid: str = None):
        self.set_cuda_properties()
        inference_request = self._inference_requests[request_uuid]
        session_id = context.get("session_id", context["sessionId"])
        direct_progress = context.get("useDirectProgressMessages", False)
        streaming_request = context.get("streamingRequest", False)
        frame_index = context["frameIndex"]
        frames_count = context["frames"]
        track_id = context["trackId"]
        video_id = context["videoId"]
        direction = context.get("direction", "forward")
        direction_n = 1 if direction == "forward" else -1
        figures = context["figures"]
        for i, figure in enumerate(figures):
            if ApiField.ID not in figure:
                figure[ApiField.ID] = i
        figures = [api.video.figure._convert_json_info(figure) for figure in figures]
        figure_id_to_object_id = {figure.id: figure.object_id for figure in figures}
        progress: sly.Progress = inference_request["progress"]
        progress_total = frames_count * len(figures) + frames_count + 1
        progress.total = progress_total
        log_extra = {
            "video_id": video_id,
            "start_frame": frame_index,
            "frames": frames_count,
            "direction": direction,
        }

        range_of_frames = [
            frame_index,
            frame_index + frames_count * direction_n,
        ]

        if self.cache.is_persistent:
            self.cache.run_cache_task_manually(
                api,
                None,
                video_id=video_id,
            )
        else:
            # if cache is not persistent, run cache task for range of frames
            self.cache.run_cache_task_manually(
                api,
                [range_of_frames if direction_n == 1 else range_of_frames[::-1]],
                video_id=video_id,
            )

        global_stop_indicatior = False

        def _add_to_inference_request(geometry, object_id, frame_index, figure_id):
            figure_info = api.video.figure._convert_json_info(
                {
                    ApiField.ID: figure_id,
                    ApiField.OBJECT_ID: object_id,
                    "meta": {"frame": frame_index},
                    ApiField.GEOMETRY_TYPE: geometry.geometry_name(),
                    ApiField.GEOMETRY: geometry.to_json(),
                }
            )
            with inference_request["lock"]:
                inference_request["pending_results"].append(figure_info)

        def _nofify_loop(q: Queue, stop_event: threading.Event):
            nonlocal global_stop_indicatior
            try:
                while True:
                    if global_stop_indicatior:
                        return
                    if inference_request["cancel_inference"]:
                        logger.info(
                            "Cancelling inference project...",
                            extra={"inference_request_uuid": request_uuid},
                        )
                        global_stop_indicatior = True
                        return
                    items = []  # (geometry, object_id, frame_index)
                    while not q.empty():
                        items.append(q.get_nowait())
                    if len(items) > 0:
                        api.logger.debug(f"got {len(items)} items to notify")
                        items_by_object_id = {}
                        for item in items:
                            items_by_object_id.setdefault(item[1], []).append(item)

                        for object_id, object_items in items_by_object_id.items():
                            frame_range = [
                                min(item[2] for item in object_items),
                                max(item[2] for item in object_items),
                            ]
                            progress.iters_done_report(len(object_items))
                            if direct_progress:
                                api.vid_ann_tool.set_direct_tracking_progress(
                                    session_id,
                                    video_id,
                                    track_id,
                                    frame_range=frame_range,
                                    progress_current=progress.current,
                                    progress_total=progress.total,
                                )
                            elif streaming_request:
                                stream_queue = self.session_stream_queue.get(session_id, None)
                                if stream_queue is None:
                                    raise RuntimeError(
                                        f"Unable to find stream queue for session {session_id}"
                                    )
                                payload = {
                                    ApiField.TRACK_ID: track_id,
                                    ApiField.VIDEO_ID: video_id,
                                    ApiField.FRAME_RANGE: frame_range,
                                    ApiField.PROGRESS: {
                                        ApiField.CURRENT: progress.current,
                                        ApiField.TOTAL: progress.total,
                                    },
                                }
                                data = {
                                    ApiField.SESSION_ID: session_id,
                                    ApiField.ACTION: "progress",
                                    ApiField.PAYLOAD: payload,
                                }
                                stream_queue.put(data)
                    else:
                        if stop_event.is_set():
                            api.logger.debug(
                                "stop event is set. returning from notify loop"
                            )
                            return
                    time.sleep(1)
            except Exception as e:
                api.logger.error("Error in notify loop: %s", str(e), exc_info=True)
                global_stop_indicatior = True
                raise

        def _upload_loop(
            q: Queue,
            notify_q: Queue,
            stop_event: threading.Event,
            stop_notify_event: threading.Event,
        ):
            nonlocal global_stop_indicatior
            try:
                while True:
                    if global_stop_indicatior:
                        return
                    if inference_request["cancel_inference"]:
                        logger.info(
                            "Cancelling inference project...",
                            extra={"inference_request_uuid": request_uuid},
                        )
                        global_stop_indicatior = True
                        return
                    items = []  # (geometry, object_id, frame_index)
                    while not q.empty():
                        items.append(q.get_nowait())
                    if len(items) > 0:
                        for item in items:
                            figure_id = uuid.uuid5(
                                namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
                            ).hex
                            _add_to_inference_request(*item, figure_id)
                            notify_q.put(item)

                    elif stop_event.is_set():
                        stop_notify_event.set()
                        return
                    time.sleep(0.1)
            except Exception as e:
                api.logger.error("Error in upload loop: %s", str(e), exc_info=True)
                stop_notify_event.set()
                global_stop_indicatior = True
                raise

        def _download_progress_cb(cnt=1):
            if cnt == 0:
                return
            progress.iters_done_report(cnt)
            if direct_progress:
                api.vid_ann_tool.set_direct_tracking_progress(
                    session_id,
                    video_id,
                    track_id,
                    frame_range=range_of_frames,
                    progress_current=progress.current,
                    progress_total=progress.total,
                )
            elif streaming_request:
                stream_queue = self.session_stream_queue.get(session_id, None)
                if stream_queue is None:
                    raise RuntimeError(
                        f"Unable to find stream queue for session {session_id}"
                    )
                payload = {
                    ApiField.TRACK_ID: track_id,
                    ApiField.VIDEO_ID: video_id,
                    ApiField.FRAME_RANGE: range_of_frames,
                    ApiField.PROGRESS: {
                        ApiField.CURRENT: progress.current,
                        ApiField.TOTAL: progress.total,
                    },
                }
                data = {
                    ApiField.SESSION_ID: session_id,
                    ApiField.ACTION: "progress",
                    ApiField.PAYLOAD: payload,
                }
                stream_queue.put(data)

        upload_queue = Queue()
        notify_queue = Queue()
        stop_upload_event = threading.Event()
        stop_notify_event = threading.Event()
        upload_thread = threading.Thread(
            target=_upload_loop,
            args=[upload_queue, notify_queue, stop_upload_event, stop_notify_event],
            daemon=True,
        )
        upload_thread.start()
        notify_thread = threading.Thread(
            target=_nofify_loop,
            args=[notify_queue, stop_upload_event],
            daemon=True,
        )
        notify_thread.start()

        inference_state = None
        api.logger.info("Start tracking.")
        try:
            temp_frames_dir = f"frames/{rand_str(10)}"
            # save frames to directory
            api.logger.debug("Saving frames to directory...", extra=log_extra)
            mkdir(temp_frames_dir, remove_content_if_exists=True)
            self.cache.download_frames_to_paths(
                api,
                video_id,
                list(
                    range(
                        frame_index,
                        frame_index + (frames_count + 1) * direction_n,
                        direction_n,
                    )
                ),
                [f"{temp_frames_dir}/{i}.jpg" for i in range(frames_count + 1)],
                progress_cb=_download_progress_cb,
            )

            api.logger.debug("Initializing model...")
            # initialize model

            if not self.video_predictor:
                self.video_predictor = build_sam2_video_predictor(
                    self.config, self.weights_path
                )

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = self.video_predictor.init_state(
                    video_path=temp_frames_dir,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                    async_loading_frames=True,
                )

            for figure in figures:
                if figure.geometry_type != sly.Bitmap.geometry_name():
                    sly.logger.warning(
                        "Only geometries of shape mask are available for tracking",
                        extra=log_extra,
                    )
                    continue
                first_frame = sly_image.read(f"{temp_frames_dir}/0.jpg")
                geometry = sly.deserialize_geometry(
                    figure.geometry_type, figure.geometry
                )
                smarttool_input = self.get_smarttool_input(figure)
                if smarttool_input is None:
                    prompt = self.generate_artificial_prompt(geometry, first_frame)
                    smarttool_input = (
                        prompt.get("bbox"),
                        prompt["point_coordinates"],
                        [],
                        True,
                    )

                # bbox - ltrb
                # points - col, row
                bbox, positive_clicks, negative_clicks, _ = smarttool_input
                if not self.use_bbox.is_switched():
                    bbox = None
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=figure.id,
                        points=positive_clicks + negative_clicks,
                        labels=[1] * len(positive_clicks) + [0] * len(negative_clicks),
                        box=bbox,
                    )

            api.logger.debug("Tracking...")
            # run propagation throughout the video
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in self.video_predictor.propagate_in_video(inference_state):
                    if global_stop_indicatior:
                        return
                    if inference_request["cancel_inference"]:
                        logger.info(
                            "Cancelling inference project...",
                            extra={"inference_request_uuid": request_uuid},
                        )
                        global_stop_indicatior = True
                        return
                    # skip first frame prediction
                    if out_frame_idx == 0:
                        continue
                    cur_frame_index = frame_index + out_frame_idx * direction_n
                    for figure_id, masks in zip(out_obj_ids, out_mask_logits):
                        masks = (masks > 0.0).cpu().numpy()
                        for i, mask in enumerate(masks):
                            sly_geometry = sly.Bitmap(mask, extra_validation=False)
                            obj_id = figure_id_to_object_id[figure_id]
                            upload_queue.put((sly_geometry, obj_id, cur_frame_index))

        except Exception as e:
            if direct_progress:
                api.vid_ann_tool.set_direct_tracking_error(
                    session_id,
                    video_id,
                    track_id,
                    message=f"An error occured during tracking. Error: {e}",
                )
            elif streaming_request:
                stream_queue = self.session_stream_queue.get(session_id, None)
                if stream_queue is None:
                    raise RuntimeError(
                        f"Unable to find stream queue for session {session_id}"
                    )
                payload = {
                    ApiField.TRACK_ID: track_id,
                    ApiField.VIDEO_ID: video_id,
                    ApiField.TYPE: "error",
                    ApiField.ERROR: {ApiField.MESSAGE: f"An error occured during tracking. Error: {e}"},
                }
                data = {
                    ApiField.SESSION_ID: session_id,
                    ApiField.ACTION: "progress",
                    ApiField.PAYLOAD: payload,
                }
                stream_queue.put(data)
            error = True
            raise
        else:
            error = False
        finally:
            if self.video_predictor is not None and inference_state is not None:
                # reset predictor state
                self.video_predictor.reset_state(inference_state)
            remove_dir(temp_frames_dir)
            stop_upload_event.set()
            if upload_thread.is_alive():
                upload_thread.join()
            stop_notify_event.set()
            if notify_thread.is_alive():
                notify_thread.join()
            if error:
                progress.message = "Error occured during tracking"
                progress.set(current=0, total=1, report=True)
            else:
                progress.message = "Ready"
                progress.set(current=0, total=1, report=True)
    
    def _setup_stream(self, session_id):
        if not hasattr(self, "session_stream_queue"):
            self.session_stream_queue = {}

        if session_id not in self.session_stream_queue:
            self.session_stream_queue[session_id] = Queue()

        def event_generator():
            q = self.session_stream_queue[session_id]
            while True:
                item = q.get()
                if item is None:
                    break
                logger.debug("streaming item: %s", item)
                yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    def serve(self):
        super().serve()
        server = self._app.get_server()

        @server.post("/smart_segmentation")
        def smart_segmentation(response: Response, request: Request):
            # 1. parse request
            # 2. download image
            # 3. make crop
            # 4. predict

            logger.debug(
                f"smart_segmentation inference: context=",
                extra={**request.state.context},
            )

            try:
                state = request.state.state
                settings = self._get_inference_settings(state)
                smtool_state = request.state.context
                api = request.state.api
                crop = smtool_state.get("crop")
                positive_clicks, negative_clicks = (
                    smtool_state["positive"],
                    smtool_state["negative"],
                )
                if len(positive_clicks) + len(negative_clicks) == 0:
                    logger.warn("No clicks received.")
                    response = {
                        "origin": None,
                        "bitmap": None,
                        "success": True,
                        "error": None,
                    }
                    return response
            except Exception as exc:
                logger.warn("Error parsing request:" + str(exc), exc_info=True)
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "400: Bad request.", "success": False}

            # collect clicks
            uncropped_clicks = [
                {**click, "is_positive": True} for click in positive_clicks
            ]
            uncropped_clicks += [
                {**click, "is_positive": False} for click in negative_clicks
            ]
            if crop:
                clicks = functional.transform_clicks_to_crop(crop, uncropped_clicks)
                is_in_bbox = functional.validate_click_bounds(crop, clicks)
                if not is_in_bbox:
                    logger.warn(f"Invalid value: click is out of bbox bounds.")
                    return {
                        "origin": None,
                        "bitmap": None,
                        "success": True,
                        "error": None,
                    }

            # download image if needed (using cache)
            app_dir = get_data_dir()
            hash_str = functional.get_hash_from_context(smtool_state)

            if hash_str not in self._inference_image_cache:
                logger.debug(f"downloading image: {hash_str}")
                try:
                    image_np = functional.download_image_from_context(
                        smtool_state,
                        api,
                        app_dir,
                        cache_load_img=self.cache.download_image,
                        cache_load_frame=self.cache.download_frame,
                        cache_load_img_hash=self.cache.download_image_by_hash,
                    )
                except Exception:
                    logger.warn("Error loading image using cache", exc_info=True)
                    image_np = api.image.download_np(smtool_state["image_id"])
                self._inference_image_cache.set(hash_str, image_np)
            else:
                logger.debug(f"image found in cache: {hash_str}")
                image_np = self._inference_image_cache.get(hash_str)

            # crop
            image_path = os.path.join(app_dir, f"{time.time()}_{rand_str(10)}.jpg")
            if isinstance(image_np, list):
                image_np = image_np[0]
            sly_image.write(image_path, image_np)

            # Prepare init_mask (only for images)
            figure_id = smtool_state.get("figure_id")
            image_id = smtool_state.get("image_id")
            if smtool_state.get("init_figure") is True and image_id is not None:
                # Download and save in Cache
                init_mask = functional.download_init_mask(api, figure_id, image_id)
                self._init_mask_cache[figure_id] = init_mask
            elif self._init_mask_cache.get(figure_id) is not None:
                # Load from Cache
                init_mask = self._init_mask_cache[figure_id]
            else:
                init_mask = None
            if init_mask is not None:
                image_info = api.image.get_info_by_id(image_id)
                init_mask = functional.bitmap_to_mask(
                    init_mask, image_info.height, image_info.width
                )
                # init_mask = functional.crop_image(crop, init_mask)
                assert init_mask.shape[:2] == image_np.shape[:2]
            settings["init_mask"] = init_mask

            self._inference_image_lock.acquire()
            try:
                # predict
                logger.debug("Preparing settings for inference request...")
                if self.use_bbox.is_switched() and crop:
                    settings["mode"] = "combined"
                else:
                    settings["mode"] = "points"
                if "image_id" in smtool_state:
                    settings["input_image_id"] = smtool_state["image_id"]
                elif "video" in smtool_state:
                    settings["input_image_id"] = hash_str
                elif "image_hash" in smtool_state:
                    settings["input_image_id"] = smtool_state["image_hash"]
                if crop:
                    settings["bbox_coordinates"] = [
                        crop[0]["y"],
                        crop[0]["x"],
                        crop[1]["y"],
                        crop[1]["x"],
                    ]
                    settings["bbox_class_name"] = "target"
                point_coordinates, point_labels = [], []
                for click in uncropped_clicks:
                    point_coordinates.append([click["x"], click["y"]])
                    if click["is_positive"]:
                        point_labels.append(1)
                    else:
                        point_labels.append(0)
                settings["point_coordinates"], settings["point_labels"] = (
                    point_coordinates,
                    point_labels,
                )
                pred_mask = self.predict(image_path, settings)[0].mask
            finally:
                logger.debug("Predict done")
                self._inference_image_lock.release()
                silent_remove(image_path)

            if pred_mask.any():
                if crop:
                    t, l, b, r = settings["bbox_coordinates"]
                    t = max(0, t)
                    l = max(0, l)
                    b = min(pred_mask.shape[0], b)
                    r = min(pred_mask.shape[1], r)
                    bitmap_data = pred_mask[t:b, l:r]
                    bitmap = sly.Bitmap(
                        bitmap_data,
                        origin=sly.PointLocation(t, l),
                        extra_validation=False,
                    )
                else:
                    bitmap_data = pred_mask
                    bitmap = sly.Bitmap(bitmap_data)
                logger.debug(f"smart_segmentation inference done!")
                response = {
                    "origin": {"x": bitmap.origin.col, "y": bitmap.origin.row},
                    "bitmap": bitmap.data_2_base64(bitmap.data),
                    "success": True,
                    "error": None,
                }
            else:
                logger.debug(f"Predicted mask is empty.")
                response = {
                    "origin": None,
                    "bitmap": None,
                    "success": True,
                    "error": None,
                }
            return response

        @server.post("/is_online")
        def is_online(response: Response, request: Request):
            response = {"is_online": True}
            return response

        @server.post("/smart_segmentation_batched")
        def smart_segmentation_batched(response: Response, request: Request):
            response_batch = {}
            data = request.state.context["data_to_process"]
            app_session_id = sly.io.env.task_id()
            for image_idx, image_data in data.items():
                image_prediction = api.task.send_request(
                    app_session_id,
                    "smart_segmentation",
                    data={},
                    context=image_data,
                )
                response_batch[image_idx] = image_prediction
            return response_batch

        @server.post("/track")
        def start_track(request: Request, task: BackgroundTasks):
            task.add_task(track, request)
            return {"message": "Tracking task started"}

        @server.post("/track-api")
        def track_api(request: Request):
            return self._track_api(request.state.api, request.state.context)

        @server.post("/track_async")
        def track_async(response: Response, request: Request):
            sly.logger.debug(
                f"'track_async' request in json format:{request.state.context}"
            )
            # check batch size
            batch_size = request.state.context.get("batch_size", self.get_batch_size())
            if self.max_batch_size is not None and batch_size > self.max_batch_size:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {
                    "message": f"Batch size should be less than or equal to {self.max_batch_size} for this model.",
                    "success": False,
                }
            inference_request_uuid = uuid.uuid5(
                namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
            ).hex
            self._on_inference_start(inference_request_uuid)
            self._inference_requests[inference_request_uuid]["lock"] = threading.Lock()
            future = self._executor.submit(
                self._handle_error_in_async,
                inference_request_uuid,
                self._track_async,
                request.state.api,
                request.state.context,
                inference_request_uuid,
            )
            end_callback = functools.partial(
                self._on_inference_end, inference_request_uuid=inference_request_uuid
            )
            future.add_done_callback(end_callback)
            sly.logger.debug(
                "Inference has scheduled from 'track_async' endpoint",
                extra={"inference_request_uuid": inference_request_uuid},
            )
            return {
                "message": "Inference has started.",
                "inference_request_uuid": inference_request_uuid,
            }

        @server.post("/pop_tracking_results")
        def pop_tracking_results(request: Request, response: Response):
            context = request.state.context
            inference_request_uuid = context.get("inference_request_uuid", None)
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                sly.logger.error("Error: 'inference_request_uuid' is required.")
                return {"message": "Error: 'inference_request_uuid' is required."}

            inference_request = self._inference_requests[inference_request_uuid]
            sly.logger.debug(
                "Pop tracking results",
                extra={
                    "inference_request_uuid": inference_request_uuid,
                    "pending_results_len": len(inference_request["pending_results"]),
                    "pending_results": [
                        {
                            "id": figure.id,
                            "object_id": figure.object_id,
                            "frame_index": figure.frame_index,
                        }
                        for figure in inference_request["pending_results"]
                    ],
                },
            )
            frame_range = context.get("frame_range", None)
            if frame_range is None:
                frame_range = context.get("frameRange", None)
            sly.logger.debug("frame_range: %s", frame_range)
            with inference_request["lock"]:
                inference_request_copy = inference_request.copy()
                inference_request_copy.pop("lock")
                inference_request_copy["progress"] = _convert_sly_progress_to_dict(
                    inference_request_copy["progress"]
                )

                if frame_range is not None:
                    inference_request_copy["pending_results"] = [
                        figure
                        for figure in inference_request_copy["pending_results"]
                        if figure.frame_index >= frame_range[0]
                        and figure.frame_index <= frame_range[1]
                    ]
                    inference_request["pending_results"] = [
                        figure
                        for figure in inference_request["pending_results"]
                        if figure.frame_index < frame_range[0]
                        or figure.frame_index > frame_range[1]
                    ]
                else:
                    inference_request["pending_results"] = []

            inference_request_copy["pending_results"] = [
                {
                    ApiField.ID: figure.id,
                    ApiField.OBJECT_ID: figure.object_id,
                    ApiField.GEOMETRY_TYPE: figure.geometry_type,
                    ApiField.GEOMETRY: figure.geometry,
                    ApiField.META: {ApiField.FRAME: figure.frame_index},
                }
                for figure in inference_request_copy["pending_results"]
            ]

            # Logging
            log_extra = _get_log_extra_for_inference_request(
                inference_request_uuid, inference_request_copy
            )
            sly.logger.debug(
                "Sending inference delta results with uuid:", extra=log_extra
            )
            return inference_request_copy

        @server.post("/stop_tracking")
        def stop_tracking(response: Response, request: Request):
            inference_request_uuid = request.state.context.get("inference_request_uuid")
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {
                    "message": "Error: 'inference_request_uuid' is required.",
                    "success": False,
                }
            inference_request = self._inference_requests[inference_request_uuid]
            inference_request["cancel_inference"] = True
            return {"message": "Inference will be stopped.", "success": True}

        @server.post("/clear_tracking_results")
        def clear_tracking_results(request: Request, response: Response):
            context = request.state.context
            inference_request_uuid = context.get("inference_request_uuid", None)
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                sly.logger.error("Error: 'inference_request_uuid' is required.")
                return {"message": "Error: 'inference_request_uuid' is required."}

            del self._inference_requests[inference_request_uuid]
            logger.debug(
                "Removed an inference request:", extra={"uuid": inference_request_uuid}
            )
            return {"success": True}

        def send_error_data(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                value = None
                try:
                    value = func(*args, **kwargs)
                except Exception as exc:
                    print("An error occured:")
                    print(traceback.format_exc())
                    request: Request = args[0]
                    context = request.state.context
                    api: sly.Api = request.state.api
                    track_id = context["trackId"]

                    api.post(
                        "videos.notify-annotation-tool",
                        data={
                            "type": "videos:tracking-error",
                            "data": {
                                "trackId": track_id,
                                "error": {"message": repr(exc)},
                            },
                        },
                    )
                return value

            return wrapper

        @mock.patch("sam2.sam2_video_predictor.tqdm", notqdm)
        @mock.patch("sam2.utils.misc.tqdm", notqdm)
        @send_error_data
        def track(request: Request):
            self._track(request.state.api, request.state.context)

        @server.post("/track_stream")
        def track_stream(request: Request):
            context = request.state.context
            logger.debug("track_stream request with context:", extra=context)
            session_id = context["session_id"]
            inference_request_uuid = uuid.uuid5(
                namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
            ).hex
            context["streamingRequest"] = True
            response = self._setup_stream(session_id)
            self._on_inference_start(inference_request_uuid)
            self._inference_requests[inference_request_uuid]["lock"] = threading.Lock()
            future = self._executor.submit(
                self._handle_error_in_async,
                inference_request_uuid,
                self._track_async,
                self.api,
                context,
                inference_request_uuid,
            )
            end_callback = functools.partial(
                self._on_inference_end, inference_request_uuid=inference_request_uuid
            )
            future.add_done_callback(end_callback)
            self.session_stream_queue[session_id].put({"sessionId": session_id, "action": "inference-started", "payload": {"inference_request_uuid": inference_request_uuid}})
            return response


m = SegmentAnything2(
    use_gui=True,
    model_dir="app_data",
    custom_inference_settings=os.path.join(root_source_path, "custom_settings.yaml"),
)


def clean_data():
    # delete app data since it is no longer needed
    sly.fs.remove_dir("prompts")
    sly.fs.remove_dir("frames")
    sly.logger.info("Successfully cleaned unnecessary app data")


if debug_session:
    if os.path.exists("prompts"):
        sly.fs.remove_dir("prompts")
    if os.path.exists("frames"):
        sly.fs.remove_dir("frames")


m.serve()
m.gui._models_table.select_row(1)
m.app.call_before_shutdown(clean_data)
