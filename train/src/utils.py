import numpy as np
import torch
import pandas as pd
import random
import supervisely as sly
import src.globals as g
from PIL import Image
import os


def get_multilabel_mask(image_np, ann):
    # extract labels
    labels = ann.labels
    multilabel_mask = np.zeros(image_np.shape, dtype=np.uint8)
    # generate multilabel mask
    for i, label in enumerate(labels):
        geometry = label.geometry
        geometry.draw(bitmap=multilabel_mask, color=[i + 1, i + 1, i + 1])
    # remove unnecessary channels
    multilabel_mask = multilabel_mask[:, :, 0]
    return multilabel_mask


def resize_image_and_mask(image, mask, size=1024, scale_proportionally=False):
    # define target width and height
    original_width, original_height = image.shape[1], image.shape[0]
    if scale_proportionally:
        scaler = min(original_width, original_height) / size
        resized_width = int(original_width / scaler)
        resized_height = int(original_height / scaler)
    else:
        resized_width = size
        resized_height = size
    # resize image
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image)
    image = torch.unsqueeze(image, 0)
    image = torch.nn.functional.interpolate(
        image, (resized_height, resized_width), mode="nearest"
    )
    image = image.squeeze().permute(1, 2, 0).numpy()
    # resize mask
    mask = torch.from_numpy(mask)
    mask = mask.view(1, 1, mask.shape[0], mask.shape[1])
    mask = torch.nn.functional.interpolate(
        mask, (resized_height, resized_width), mode="nearest"
    )
    mask = mask.squeeze().numpy()
    return image, mask


def generate_artificial_prompts(
    multilabel_mask,
    image_np,
    n_points,
    use_bbox=False,
):
    labels = np.unique(multilabel_mask)[1:]  # skip background label
    prompts = {"point_coordinates": [], "point_labels": []}
    if use_bbox:
        prompts["bboxes"] = []

    for label in labels:
        # generate point prompts
        mask = (multilabel_mask == label).astype(np.uint8)
        row_indexes, col_indexes = np.where(mask)
        point_coordinates = []
        for i in range(n_points):
            idx = np.random.randint(0, len(row_indexes))
            row_index, col_index = row_indexes[idx], col_indexes[idx]
            point_coordinates.append([col_index, row_index])
        prompts["point_coordinates"].extend(point_coordinates)
        prompts["point_labels"].extend(np.ones((n_points,)))
        if use_bbox:
            # generate box prompt
            left, top, right, bottom = [
                np.min(col_indexes),
                np.min(row_indexes),
                np.max(col_indexes),
                np.max(row_indexes),
            ]
            # apply padding
            padding = 10
            padded_left = left - padding
            padded_top = top - padding
            padded_right = right + padding
            padded_bottom = bottom + padding
            # check if padded bbox is not out of image bounds
            image_h, image_w = image_np.shape[0], image_np.shape[1]
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
            prompts["bboxes"].append(bbox)

    return prompts


def generate_prompts_for_batch(masks, images, n_points):
    (
        batch_images,
        batch_masks,
        batch_points,
        batch_labels,
    ) = ([], [], [], [])
    for image, mask in zip(images, masks):
        image = image.numpy()
        mask = mask.numpy()
        batch_images.append(image)
        image_prompts = generate_artificial_prompts(
            mask,
            image,
            n_points,
        )
        batch_points.append(image_prompts["point_coordinates"])
        batch_labels.append(image_prompts["point_labels"])
        mask = np.where(mask != 0, 1, 0).astype(np.uint8)
        batch_masks.append(mask)
    return batch_images, batch_masks, batch_points, batch_labels


def generate_history_df(
    loss_history,
    iou_history,
    mode,
):
    if mode == "train":
        df_data = {
            "train loss": loss_history,
            "train iou": iou_history,
        }
    elif mode == "val":
        df_data = {
            "val loss": loss_history,
            "val iou": iou_history,
        }
    epochs_list = [i for i in range(1, len(loss_history) + 1)]
    df = pd.DataFrame(data=df_data, index=epochs_list)
    return df


def generate_predictions(
    val_set, predictor, project_meta, min_points, max_points, pbar, n_pairs=8
):
    annotated_img_paths = []
    save_dir = os.path.join(g.train_artifacts_dir, "predictions")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    items = random.choices(val_set, k=n_pairs)
    with pbar(message="Generating model predictions...", total=n_pairs) as preds_pbar:
        for i, item in enumerate(items):
            # get image and annotation paths
            img_path = item.img_path
            gt_ann_path = item.ann_path
            # draw ground truth labels on image
            img = sly.image.read(img_path)
            gt_img = img.copy()
            gt_ann = sly.Annotation.load_json_file(gt_ann_path, project_meta)
            gt_ann.draw_pretty(gt_img, thickness=1, color=[255, 0, 0])
            # save labeled ground truth image
            gt_img = Image.fromarray(gt_img)
            gt_img_path = os.path.join(save_dir, f"gt_{i}.jpg")
            gt_img.save(gt_img_path)
            # get multilabel mask
            multilabel_mask = get_multilabel_mask(img, gt_ann)
            # generate prompt
            n_points = random.randint(min_points, max_points)
            prompt = generate_artificial_prompts(multilabel_mask, img, n_points)
            # apply model to image
            point_coordinates = np.array(prompt["point_coordinates"])
            point_labels = np.array(prompt["point_labels"])
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    predictor.set_image(img)
                    if len(point_labels) > 1:
                        masks, _, _ = predictor.predict(
                            point_coords=point_coordinates,
                            point_labels=point_labels,
                            multimask_output=False,
                        )
                        mask = masks[0]
                    else:
                        masks, scores, logits = predictor.predict(
                            point_coords=point_coordinates,
                            point_labels=point_labels,
                            multimask_output=True,
                        )
                        max_score_ind = np.argmax(scores)
                        mask = masks[max_score_ind]
            # generate sly annotation from predicted mask
            bitmap = sly.Bitmap(mask)
            obj_class = gt_ann.labels[0].obj_class
            label = sly.Label(bitmap, obj_class)
            img_height, img_width = img.shape[:2]
            pred_ann = sly.Annotation(img_size=[img_height, img_width], labels=[label])
            pred_img = img.copy()
            pred_ann.draw_pretty(pred_img, thickness=1, color=[255, 0, 0])
            # save labeled predicted image
            pred_img = Image.fromarray(pred_img)
            pred_img_path = os.path.join(save_dir, f"pred_{i}.jpg")
            pred_img.save(pred_img_path)
            annotated_img_paths.append([gt_img_path, pred_img_path])
            preds_pbar.update()
    pbar.hide()
    return annotated_img_paths


def clip_gradients(predictor, method, threshold):
    if method == "by norm":
        torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=threshold)
    elif method == "by value":
        torch.nn.utils.clip_grad_value_(
            predictor.model.parameters(), clip_value=threshold
        )
