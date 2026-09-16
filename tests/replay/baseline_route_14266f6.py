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
        self.process_volume = smtool_state.get("volume") is not None
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
            if "pcd_related_image_id" in smtool_state:
                image_np = api.pointcloud.download_related_image(
                    smtool_state["pcd_related_image_id"]
                )
            else:
                image_np = api.image.download_np(smtool_state["image_id"])
        self._inference_image_cache.set(hash_str, image_np)
    else:
        logger.debug(f"image found in cache: {hash_str}")
        image_np = self._inference_image_cache.get(hash_str)

    # crop
    image_path = os.path.join(
        app_dir, f'{str(time.time()).replace(".", "_")}_{rand_str(10)}.jpg'
    )
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
        if self.process_volume:
            volume_id = smtool_state.get("volume").get("volume_id")
            volume_plane = get_plane_name(
                smtool_state.get("volume").get("normal")
            )
            slice_idx = smtool_state.get("volume").get("slice_index")
            settings["input_image_id"] = (
                f"{volume_id}_{volume_plane}_{slice_idx}"
            )
        else:
            if "image_id" in smtool_state:
                settings["input_image_id"] = smtool_state["image_id"]
            elif "video" in smtool_state:
                settings["input_image_id"] = hash_str
            elif "image_hash" in smtool_state:
                settings["input_image_id"] = smtool_state["image_hash"]
            elif "pcd_related_image_id" in smtool_state:
                settings["input_image_id"] = smtool_state[
                    "pcd_related_image_id"
                ]
        if crop:
            settings["bbox_coordinates"] = [
                crop[0]["y"],
                crop[0]["x"],
                crop[1]["y"] + 1,
                crop[1]["x"] + 1,
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
