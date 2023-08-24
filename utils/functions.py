
import os
import numpy as np
import json
import cv2
import torch


def draw_points(img, points, labels, color_obj, color_bg):
    for point, label in zip(points, labels):
        x, y = point[0], point[1]
        if label == 1:
            color = color_obj
        else:
            color = color_bg
        cv2.circle(img, (x, y), 3, color, -1)


def load_embedding(sam_predictor, embeddings_path, img_name, image, device):
    # Change extension to .pt
    filename = os.path.splitext(img_name)[0] + ".pt"
    embedding_file = os.path.join(embeddings_path, filename)

    if not os.path.exists(embedding_file):
        return False
    else:
        print(f"Embedding found in {embedding_file}")
        embedding = torch.load(embedding_file)

        sam_predictor.original_size = image.shape[:2]

        # Transform the image to the form expected by the model
        input_image = sam_predictor.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        sam_predictor.input_size = tuple(input_image_torch.shape[-2:])
        sam_predictor.features = embedding
        sam_predictor.is_image_set = True
        return True

def save_embedding(embedding, out_path, img_name):
    # Change extension to .pt
    filename = os.path.splitext(img_name)[0] + ".pt"

    # Save embedding
    out_file = os.path.join(out_path, filename)
    torch.save(embedding, out_file)
    print(f"Saving embedding in {out_file}")

def save_masks(masks_data, out_path, img_name):
    # Change extension to png
    filename = os.path.splitext(img_name)[0] + ".png"

    if len(masks_data) > 0:
        first_mask = masks_data[0]["mask"]
        print("First mask shape:", first_mask.shape)
        masks_canvas = np.zeros((first_mask.shape[:2]), dtype=np.uint8)

        # loop through all the masks and draw them in a single canvas using the idx as value
        for idx, masks_datum in enumerate(masks_data):
            mask_u8 = masks_datum["mask"]
            if mask_u8 is not None:
                masks_canvas[mask_u8 == 255] = idx + 1

        # Save mask
        out_file = os.path.join(out_path, img_name)
        print(f"Saving Mask in {out_file}")
        cv2.imwrite(out_file, masks_canvas)
        print("Masks saved in file!")

def load_points_and_labels(dir_path, img_name, predictor, scale):
    # Change extension to json
    filename = os.path.splitext(img_name)[0] + ".json"
    file_path = os.path.join(dir_path, filename)

    data = []
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)

    for dic in data:
        points = dic["sam_points"]
        labels = dic["sam_labels"]

        mask, scores, logits = predictor.predict(
            point_coords=np.array(points) / scale,
            point_labels=np.array(labels),
            multimask_output=False,
        )
        mask = np.squeeze(mask)  # remove leading dimension
        mask_u8 = mask.astype(np.uint8) * 255
        dic["mask"] = mask_u8
    return data


def save_points_and_labels(data, out_path, img_name):
    # Change extension to json
    filename = os.path.splitext(img_name)[0] + ".json"
    out_file_path = os.path.join(out_path, filename)

    # Remove the masks so we can save in json
    for dic in data:
        dic.pop("mask", None)

    with open(out_file_path, "w") as file:
        json.dump(data, file)

def rescale_points(points, scale):
    points_scaled = [[int(p[0] / scale), int(p[1] / scale)] for p in points]
    return points_scaled

def scale_points(points, scale):
    points_scaled = [[int(p[0] * scale), int(p[1] * scale)] for p in points]
    return points_scaled

def delete_last_point(points, labels):
    if points:
        points.pop()
    if labels:
        labels.pop()
    return points, labels

def load_mask_data(masks_data, index, scale):
    print(f"Masks_data size inside load_mask_data: {len(masks_data)}")
    print(f"Index: {index}")

    if len(masks_data) <= index:
        points = []
        labels = []
    else:
        points = masks_data[index]["sam_points"]
        labels = masks_data[index]["sam_labels"]
        points = scale_points(points, scale)
        
    return points, labels

# Display all the masks in blue and the current one in red
def display_all_masks(masks_data, current_index, current_mask, image, scale):
    image_with_masks = np.copy(image)

    mask_rgb = np.zeros((current_mask.shape[:2] + (3,)), dtype=np.uint8)  # Create an initial RGB image with zeros
    if len(masks_data) > 0:
        masks_canvas = np.zeros((current_mask.shape[:2]), dtype=np.uint8)
        # loop through all the masks and draw them in a single canvas
        for idx, masks_datum in enumerate(masks_data):
            if idx == current_index:
                continue

            #print(f"idx: {idx}")
            mask_u8 = masks_datum["mask"]
            if mask_u8 is not None:
                masks_canvas = np.bitwise_or(masks_canvas, mask_u8)

        mask_rgb[:, :, 0] = masks_canvas  # Set the blue channel to the mask values
    # draw the current mask
    mask_rgb[:, :, 2] = current_mask  # Set the red channel to the mask values

    mask_rgb_resized = cv2.resize(mask_rgb, None, fx=scale, fy=scale)
    image_with_masks = cv2.addWeighted(image, 0.8, mask_rgb_resized, 0.2, 0)
    return image_with_masks


# Display the saved masks
def display_saved_masks(masks_data, image, scale):
    image_with_masks = np.copy(image)

    if len(masks_data) > 0:
        first_mask = masks_data[0]["mask"]
        mask_rgb = np.zeros((first_mask.shape[:2] + (3,)), dtype=np.uint8)  # Create an initial RGB image with zeros
        masks_canvas = np.zeros((first_mask.shape[:2]), dtype=np.uint8)
        # loop through all the masks and draw them in a single canvas
        for idx, masks_datum in enumerate(masks_data):
            mask_u8 = masks_datum["mask"]
            if mask_u8 is not None:
                masks_canvas = np.bitwise_or(masks_canvas, mask_u8)

        mask_rgb[:, :, 0] = masks_canvas  # Set the blue channel to the mask values

        mask_rgb_resized = cv2.resize(mask_rgb, None, fx=scale, fy=scale)
        image_with_masks = cv2.addWeighted(image, 0.8, mask_rgb_resized, 0.2, 0)
    else:
        return image
    return image_with_masks
