import os
import numpy as np
import torch
import cv2
import threading
import time
from cacheout import Cache
from dotenv import load_dotenv
from typing import Literal
from typing import List, Any, Dict
from fastapi import Response, Request, status, BackgroundTasks
from pathlib import Path
from cachetools import LRUCache
import supervisely as sly
from supervisely.imaging.color import generate_rgb
from supervisely.nn.inference.interactive_segmentation import functional
from supervisely.sly_logger import logger
from supervisely.imaging import image as sly_image
from supervisely.io.fs import silent_remove
from supervisely._utils import rand_str, is_debug_with_sly_net
from supervisely.app.content import get_data_dir
import supervisely.app.development as sly_app_development
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import functools
import json
import traceback
from PIL import Image
import mock


load_dotenv("supervisely.env")
load_dotenv("debug.env")
api = sly.Api()
root_source_path = str(Path(__file__).parents[1])
debug_session = bool(os.environ.get("DEBUG_SESSION", False))
model_data_path = os.path.join(root_source_path, "models", "models.json")


class SegmentAnything2(sly.nn.inference.PromptableSegmentation):

    def support_custom_models(self):
        return False

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

    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        # get weights path and config
        self.weights_path, self.config = self.get_weights_path_and_config()
        # build model
        self.sam = build_sam2(self.config, self.weights_path, device=device)
        # load model on device
        if device != "cpu":
            if device == "cuda":
                torch.cuda.set_device(0)
            else:
                torch.cuda.set_device(int(device[-1]))
            torch_device = torch.device(device)
            self.sam.to(device=torch_device)
        else:
            self.sam.to(device=device)
        # build predictor
        self.predictor = SAM2ImagePredictor(self.sam)
        # define class names
        self.class_names = ["target_mask"]
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
        info["videos_support"] = False
        info["async_video_inference_support"] = False
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
                self.predictor.set_image(input_image)
                self.model_cache.set(
                    settings["input_image_id"],
                    {
                        "features": self.predictor._features,
                        "original_size": self.predictor._orig_hw,
                    },
                )
            else:
                cached_data = self.model_cache.get(settings["input_image_id"])
                self.predictor._features = cached_data["features"]
                self.predictor._orig_hw = cached_data["original_size"]

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionMask]:
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
            )
            masks = mask_generator.generate(input_image)
            for i, mask in enumerate(masks):
                class_name = "object_" + str(i)
                # add new class to model meta if necessary
                if not self._model_meta.get_obj_class(class_name):
                    color = generate_rgb(self.mask_colors)
                    self.mask_colors.append(color)
                    self.class_names.append(class_name)
                    new_class = sly.ObjClass(class_name, sly.Bitmap, color)
                    self._model_meta = self._model_meta.add_obj_class(new_class)
                # get predicted mask
                mask = mask["segmentation"]
                predictions.append(
                    sly.nn.PredictionMask(class_name=class_name, mask=mask)
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
            if settings["points_class_name"]:
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
            masks, _, _ = self.predictor.predict(
                point_coords=point_coordinates,
                point_labels=point_labels,
                multimask_output=False,
            )
            mask = masks[0]
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
            init_mask = settings["init_mask"]
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
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coordinates,
                    point_labels=point_labels,
                    box=bbox_coordinates[None, :],
                    mask_input=mask_input[None, :, :],
                    multimask_output=False,
                )
            elif init_mask is not None:
                # transform
                mask_input = self.predictor.transform.apply_image(init_mask)
                # pad
                h, w = mask_input.shape[:2]
                padh = self.predictor.model.image_encoder.img_size - h
                padw = self.predictor.model.image_encoder.img_size - w
                mask_input = np.pad(mask_input, ((0, padh), (0, padw)))
                # downscale to 256x256
                mask_input = cv2.resize(
                    mask_input, (256, 256), interpolation=cv2.INTER_LINEAR
                )
                # put values
                mask_input = mask_input.astype(float)
                mask_input[mask_input > 0] = 20
                mask_input[mask_input <= 0] = -20
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
                    multimask_output=False,
                )
            # save bbox ccordinates and mask to cache
            if settings["input_image_id"] in self.model_cache:
                image_id = settings["input_image_id"]
                cached_data = self.model_cache.get(image_id)
                cached_data["previous_bbox"] = bbox_coordinates
                cached_data["mask_input"] = logits[0]
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

    def generate_artificial_prompt(self, bitmap):
        mask = bitmap.data
        origin_row, origin_col = bitmap.origin.row, bitmap.origin.col
        row_indexes, col_indexes = np.where(mask)
        point_coordinates, point_labels = [], []
        for i in range(4):
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
        return prompt

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
                crop = smtool_state["crop"]
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
                image_np = functional.download_image_from_context(
                    smtool_state,
                    api,
                    app_dir,
                    cache_load_img=self.cache.download_image,
                    cache_load_frame=self.cache.download_frame,
                    cache_load_img_hash=self.cache.download_image_by_hash,
                )
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
                settings["mode"] = "combined"
                if "image_id" in smtool_state:
                    settings["input_image_id"] = smtool_state["image_id"]
                elif "video" in smtool_state:
                    settings["input_image_id"] = hash_str
                elif "image_hash" in smtool_state:
                    settings["input_image_id"] = smtool_state["image_hash"]
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
                if "video" in smtool_state:
                    # save prompt to json file
                    video_id = smtool_state["video"]["video_id"]
                    if not os.path.exists(f"prompts/{video_id}"):
                        os.makedirs(f"prompts/{video_id}")
                    bbox_str = (
                        f"{crop[0]['y']}-{crop[0]['x']}-{crop[1]['y']}-{crop[1]['x']}"
                    )
                    prompt = {
                        "0": {
                            bbox_str: {
                                "point_coordinates": point_coordinates,
                                "point_labels": point_labels,
                            },
                        }
                    }
                    prompt_filepath = f"prompts/{video_id}/point_coordinates.json"
                    if os.path.exists(prompt_filepath):
                        with open(prompt_filepath, "r") as file:
                            previous_prompt = json.load(file)
                        if "0" in previous_prompt:
                            if not bbox_str in previous_prompt["0"]:
                                prompt["0"] = {
                                    **prompt["0"],
                                    **previous_prompt["0"],
                                }
                        else:
                            prompt = {**previous_prompt, **prompt}

                    with open(prompt_filepath, "w") as file:
                        json.dump(prompt, file)

                pred_mask = self.predict(image_path, settings)[0].mask
            finally:
                logger.debug("Predict done")
                self._inference_image_lock.release()
                silent_remove(image_path)

            if pred_mask.any():
                bitmap = sly.Bitmap(pred_mask)
                # crop bitmap
                bitmap = bitmap.crop(sly.Rectangle(*settings["bbox_coordinates"]))[0]
                # adapt bitmap to crop coordinates
                bitmap_data = bitmap.data
                bitmap_origin = sly.PointLocation(
                    bitmap.origin.row - crop[0]["y"],
                    bitmap.origin.col - crop[0]["x"],
                )
                bitmap = sly.Bitmap(data=bitmap_data, origin=bitmap_origin)

                if "video" in smtool_state:
                    bitmap_center = self.get_bitmap_center(bitmap)
                    bitmap_center_str = f"{bitmap_center[0]}-{bitmap_center[1]}"
                    bitmap_data = {
                        "0": {
                            bitmap_center_str: bbox_str,
                        }
                    }
                    bitmap_data_path = f"prompts/{video_id}/bitmap2bbox.json"
                    if os.path.exists(bitmap_data_path):
                        with open(bitmap_data_path, "r") as file:
                            previous_bitmap_data = json.load(file)
                        if "0" in previous_bitmap_data:
                            if not bitmap_center_str in previous_bitmap_data["0"]:
                                bitmap_data["0"] = {
                                    **bitmap_data["0"],
                                    **previous_bitmap_data["0"],
                                }
                    with open(bitmap_data_path, "w") as file:
                        json.dump(bitmap_data, file)

                bitmap_origin, bitmap_data = functional.format_bitmap(bitmap, crop)
                logger.debug(f"smart_segmentation inference done!")
                response = {
                    "origin": bitmap_origin,
                    "bitmap": bitmap_data,
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

        def notqdm(iterable, *args, **kwargs):
            """
            replacement for tqdm that just passes back the iterable
            useful to silence `tqdm` in tests
            """
            return iterable

        @mock.patch("sam2.sam2_video_predictor.tqdm", notqdm)
        @mock.patch("sam2.utils.misc.tqdm", notqdm)
        @send_error_data
        def track(request: Request):
            sly.logger.info("Starting tracking process...")
            # get input data
            mode = "user clicks"
            context = request.state.context
            api = request.state.api
            video_id = context["videoId"]
            track_id = context["trackId"]
            frames_dir = f"frames/{video_id}"
            prompt_path = f"prompts/{video_id}/point_coordinates.json"
            bitmap_data_path = f"prompts/{video_id}/bitmap2bbox.json"
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)
            n_frames = context["frames"]
            start_frame = context["frameIndex"]
            pbar_length = 2 * (n_frames + 1) + n_frames
            pbar_pos = 0
            fstart = start_frame
            fend = start_frame + n_frames
            # check if prompt exist (if we are working with masks created by model)
            if not os.path.exists(prompt_path):
                mode = "artificial clicks"
            # download frames
            for i in range(n_frames + 1):
                frame_index = i + start_frame
                frame_np = api.video.frame.download_np(video_id, frame_index)
                frame_img = Image.fromarray(frame_np)
                frame_img.save(f"{frames_dir}/{i}.jpeg")
                pbar_pos += 1
                api.video.notify_progress(
                    track_id,
                    video_id,
                    fstart,
                    fend,
                    pbar_pos,
                    pbar_length,
                )
            # get input prompts
            if mode == "user clicks":
                with open(prompt_path, "r") as file:
                    prompts = json.load(file)
                frame_prompts = prompts["0"]
            if mode == "user clicks":
                with open(bitmap_data_path, "r") as file:
                    bitmap_data = json.load(file)
                bitmap_frame_data = bitmap_data["0"]
            # match frame prompts with figures
            figure_ids = context["figureIds"]
            figures_data = []
            fig_id2_obj_id = {}
            for figure_id in figure_ids:
                figure = self.api.video.figure.get_info_by_id(figure_id)
                if figure.geometry_type != "bitmap":
                    sly.logger.warn(
                        "Only geometries of shape mask are available for tracking"
                    )
                    continue
                object_id = figure.object_id
                fig_id2_obj_id[figure_id] = object_id
                geometry = sly.deserialize_geometry(
                    figure.geometry_type, figure.geometry
                )
                if mode == "user clicks":
                    bitmap_center = self.get_bitmap_center(geometry)
                    bitmap_center_str = f"{bitmap_center[0]}-{bitmap_center[1]}"
                    bbox_str = bitmap_frame_data[bitmap_center_str]
                    figure_prompt = frame_prompts[bbox_str]
                else:
                    figure_prompt = self.generate_artificial_prompt(geometry)
                figure_data = {
                    "figure_id": figure_id,
                    "object_id": object_id,
                    "figure_prompt": figure_prompt,
                }
                figures_data.append(figure_data)
            # initialize model
            video_predictor = build_sam2_video_predictor(self.config, self.weights_path)
            inference_state = video_predictor.init_state(video_path=frames_dir)
            # pass input prompts to the model
            for figure_data in figures_data:
                fig_id = figure_data["figure_id"]
                fig_prompt = figure_data["figure_prompt"]
                point_coordinates = fig_prompt["point_coordinates"]
                point_labels = fig_prompt["point_labels"]
                _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=fig_id,
                    points=point_coordinates,
                    labels=point_labels,
                )
            # run propagation throughout the video
            video_segments = {}  # per-frame segmentation results
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in video_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                pbar_pos += 1
                api.video.notify_progress(
                    track_id,
                    video_id,
                    fstart,
                    fend,
                    pbar_pos,
                    pbar_length,
                )
            # upload segmentation results to the platform
            for i in range(1, n_frames + 1):
                frame_result = video_segments[i]
                frame_idx = i + start_frame
                for fig_id, masks in frame_result.items():
                    for mask in masks:
                        obj_id = fig_id2_obj_id[fig_id]
                        geometry = sly.Bitmap(mask)
                        api.video.figure.create(
                            video_id,
                            obj_id,
                            frame_idx,
                            geometry.to_json(),
                            geometry.geometry_name(),
                            track_id,
                        )
                pbar_pos += 1
                api.video.notify_progress(
                    track_id,
                    video_id,
                    fstart,
                    fend,
                    pbar_pos,
                    pbar_length,
                )
            # reset predictor state
            video_predictor.reset_state(inference_state)
            if os.path.exists(frames_dir):
                sly.fs.clean_dir(frames_dir)
            if os.path.exists(f"prompts/{video_id}"):
                sly.fs.clean_dir(f"prompts/{video_id}")
            sly.logger.info("Successfully finished tracking process")


if is_debug_with_sly_net():
    team_id = sly.env.team_id()
    original_dir = os.getcwd()
    sly_app_development.supervisely_vpn_network(action="up")
    task = sly_app_development.create_debug_task(
        team_id, port="8000", update_status=True
    )
    os.chdir(original_dir)

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
