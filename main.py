"""
Project Name: Auto-Seg-Annotation
Description: Automatic Segmentation Annotation tool based on SAM (Segment Anything Model)
Author: Juan Terven
Date: August 2023
License: MIT
Contact: jrterven@hotmail.com
"""
import os
import argparse
import numpy as np
import sys
import cv2
import json
import torch
import time
from utils import functions as fn
from constants import COLOR_OBJ, COLOR_BG
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
except:
    print("SAM is not installed. Please install with:")
    print("pip install git+https://github.com/facebookresearch/segment-anything.git")
    sys.exit(1)


# Function to handle mouse events like left click
def click_event(event, x, y, flags, param):
    global SCALED_SAM_POINTS, SAM_LABELS, IMG_CANVAS

    if event == cv2.EVENT_LBUTTONDOWN:  # Check if left mouse button was clicked
        for i, (px, py) in enumerate(SCALED_SAM_POINTS):
            distance = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
            if distance <= 3:
                del SCALED_SAM_POINTS[i]
                del SAM_LABELS[i]
                print(f"Removed point at ({px}, {py})")
                break
        else:
            cv2.circle(IMG_CANVAS, (x, y), 3, COLOR_OBJ, -1)
            cv2.imshow('Image', IMG_CANVAS)
            SCALED_SAM_POINTS.append([x, y])
            SAM_LABELS.append(1)
        print(f"{len(SCALED_SAM_POINTS)} points:", SCALED_SAM_POINTS)

    if event == cv2.EVENT_RBUTTONDOWN:  # Check if right mouse button was clicked
        cv2.circle(IMG_CANVAS, (x, y), 3, COLOR_BG, -1)
        cv2.imshow('Image', IMG_CANVAS)
        SCALED_SAM_POINTS.append([x, y])
        SAM_LABELS.append(0)

        print(f"{len(SCALED_SAM_POINTS)} points:", SCALED_SAM_POINTS)



def main(main_args):
    global SCALED_SAM_POINTS, SAM_LABELS, IMG_CANVAS

    project_directory = main_args["project_directory"]
    fast_mode = main_args["fast_mode"]
    checkpoint_name = main_args["checkpoint_name"]
    checkpoint_path = main_args["checkpoint_path"]
    model_type = main_args["model_type"]
    device = main_args["device"]

    images_path = os.path.join(project_directory, "images")
    masks_path = os.path.join(project_directory, "masks")
    emb_path = os.path.join(project_directory, "embeddings")

    # Load images
    image_names = os.listdir(images_path)
    num_images = len(image_names)
    print(f"Found {len(image_names)} images.")

    if fast_mode:
        # Load FastSAM model
        print("Loading FastSAM")    
        fsam_model = FastSAM(checkpoint_path)
    else:
        # Loading SAM model
        print("Loading SAM ...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device=device)
        predictor = SamPredictor(sam)

    exit_flag = False
    img_idx = 0
    show_mask = True
    show_points = True
    size_scale = 1.0

    # Loop through each image
    while True:
        current_mask_index = 0
        image_name = image_names[img_idx]
        image_path = os.path.join(images_path, image_name)
        img_orig = cv2.imread(image_path)
        h_orig, w_orig, _ = img_orig.shape
        print(f"Original Image: {h_orig}x{w_orig}")

        # Image used for visualization in case the original is too big or too small
        img_resized = cv2.resize(img_orig, None, fx=size_scale, fy=size_scale)
        IMG_CANVAS = np.copy(img_resized)
        h_canv, w_canv, _ = IMG_CANVAS.shape
        print(f"Vis Image: {h_canv}x{w_canv}")
        cv2.imshow('Image', IMG_CANVAS)

        print(f"size_scale: {size_scale}")

        mask_rgb = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)  # Create an initial RGB image with zeros
        mask_u8 = np.zeros((h_orig, w_orig), dtype=np.uint8)  # Create an initial RGB image with zeros
        old_num_points = 0 # keep track if the number of points changed

        if fast_mode:
            # Run inference on an image
            start_time = time.time()
            model_results = fsam_model(image_path, 
                                        device=device,
                                        retina_masks=True,
                                        imgsz=1024,
                                        conf=0.4,
                                        iou=0.9)
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time = end_time - start_time
            print("Elapsed time: ", elapsed_time)

            predictor = FastSAMPrompt(image_path, model_results,
                                           device=device)
        else:
            print("Loading SAM embedding...")
            got_embedding = fn.load_embedding(predictor, emb_path, image_name, img_orig, device)
            if not got_embedding:
                print("Computing SAM embedding...")
                start_time = time.time()
                predictor.set_image(img_orig)
                end_time = time.time()

                # Calculate elapsed time
                elapsed_time = end_time - start_time
                print("Elapsed time: ", elapsed_time)

                # Save the embedding for next time
                embedding = predictor.get_image_embedding()
                fn.save_embedding(embedding, emb_path, image_name)

        SCALED_SAM_POINTS = []
        SAM_LABELS = []
        MASKS_DATA = fn.load_points_and_labels(emb_path, image_name,
                                               predictor, fast_mode, 1.0)

        current_mask_index = len(MASKS_DATA)
        print(f"Found {current_mask_index} masks in current image.")
        print(f"Mask Index = {current_mask_index}.")

        if len(MASKS_DATA) > 0:
            SCALED_SAM_POINTS, SAM_LABELS = fn.load_mask_data(MASKS_DATA, current_mask_index, size_scale)

        old_num_points = 0
        IMG_CANVAS = fn.display_saved_masks(MASKS_DATA, img_resized, size_scale)
        cv2.imshow('Image', IMG_CANVAS)

        # Loop until we change image
        while True:
            IMG_CANVAS = np.copy(img_resized)
            if len(SCALED_SAM_POINTS) != old_num_points and len(SCALED_SAM_POINTS) > 0:
                if SCALED_SAM_POINTS:
                    mask_u8 = fn.predict_masks(predictor, SCALED_SAM_POINTS,
                                            np.array(SAM_LABELS), size_scale, fast_mode)
                else:
                    mask_u8 = np.zeros((h_orig, w_orig), dtype=np.uint8)
                IMG_CANVAS = fn.display_all_masks(MASKS_DATA, current_mask_index, mask_u8, img_resized, size_scale)
                fn.draw_points(IMG_CANVAS, SCALED_SAM_POINTS, SAM_LABELS, COLOR_OBJ, COLOR_BG)
                cv2.imshow('Image', IMG_CANVAS)

            if  len(SCALED_SAM_POINTS) != old_num_points and len(SCALED_SAM_POINTS) == 0:
                IMG_CANVAS = fn.display_all_masks(MASKS_DATA, current_mask_index, mask_u8, img_resized, size_scale)
                cv2.imshow('Image', IMG_CANVAS)

            old_num_points = len(SCALED_SAM_POINTS)
            key = cv2.waitKeyEx(100)

            # Press "q" or ESC to exit program,
            if key == ord("q") or key == ord("Q") or key == 27:
                # Save current data
                fn.save_masks(MASKS_DATA, masks_path, image_name)
                fn.save_points_and_labels(MASKS_DATA, emb_path, image_name)

                exit_flag = True
                break
            # Press "n" to go to the next image without saving the mask and points
            elif key == ord('n') or key == ord('N'):
                # Save current data
                fn.save_masks(MASKS_DATA, masks_path, image_name)
                fn.save_points_and_labels(MASKS_DATA, emb_path, image_name)

                img_idx += 1
                if img_idx >= num_images:
                    img_idx = num_images - 1
                break
            # Press "b" to go to back to the previous image without saving the mask and points
            elif key == ord('b') or key == ord('B'):
                # Save current data
                fn.save_masks(MASKS_DATA, masks_path, image_name)
                fn.save_points_and_labels(MASKS_DATA, emb_path, image_name)

                img_idx -= 1
                if img_idx < 0:
                    img_idx = 0
                break
            # Press "c" to clear all the points
            elif key == ord('c') or key == ord('C'):
                SCALED_SAM_POINTS = []
                SAM_LABELS = []
            # Press "s" to save mask and points
            elif key == ord('s') or key == ord('S'):
                fn.save_masks(MASKS_DATA, masks_path, image_name)
                fn.save_points_and_labels(MASKS_DATA, emb_path, image_name)
            # Press "d" to delete the last point
            elif key == ord('d') or key == ord('D'):
                SCALED_SAM_POINTS, SAM_LABELS = fn.delete_last_point(SCALED_SAM_POINTS, SAM_LABELS)
            # Press "m" to show or hide current mask from visualization
            elif key == ord('m'):
                show_mask = not show_mask
                IMG_CANVAS = np.copy(img_resized)
                if show_mask:
                    IMG_CANVAS = fn.display_all_masks(MASKS_DATA, current_mask_index, mask_u8, IMG_CANVAS, size_scale)
                if show_points:
                    fn.draw_points(IMG_CANVAS, SCALED_SAM_POINTS, SAM_LABELS, COLOR_OBJ, COLOR_BG)
                cv2.imshow('Image', IMG_CANVAS)
            # Press "M" to show or hide all the masks from visualization
            elif key == ord('M'):
                show_mask = not show_mask

                IMG_CANVAS = np.copy(img_resized)
                if show_mask:
                    IMG_CANVAS = fn.display_all_masks(MASKS_DATA, current_mask_index, mask_u8, IMG_CANVAS, size_scale)
                if show_points:
                    fn.draw_points(IMG_CANVAS, SCALED_SAM_POINTS, SAM_LABELS, COLOR_OBJ, COLOR_BG)
                cv2.imshow('Image', IMG_CANVAS)
            # Press "p" to remove points from visualization
            elif key == ord('p') or key == ord('P'):
                show_points = not show_points

                IMG_CANVAS = np.copy(img_resized)
                if show_mask:
                    IMG_CANVAS = fn.display_all_masks(MASKS_DATA, current_mask_index, mask_u8, IMG_CANVAS, size_scale)
                if show_points:
                    fn.draw_points(IMG_CANVAS, SCALED_SAM_POINTS, SAM_LABELS, COLOR_OBJ, COLOR_BG)
                cv2.imshow('Image', IMG_CANVAS)
            # Press enter to save the current mask
            elif key == 13:
                if len(SCALED_SAM_POINTS) > 0:
                    points = fn.rescale_points(SCALED_SAM_POINTS, size_scale)
                    masks_dict = {"sam_points": points, "sam_labels": SAM_LABELS, "mask": mask_u8}
                    if current_mask_index >= len(MASKS_DATA):
                        MASKS_DATA.append(masks_dict)
                    else:
                        MASKS_DATA[current_mask_index] = masks_dict
                # In case no points, 
                # e.g. the user pressed enter with no points
                # e.g. the user deleted all the points and press enter
                else:
                    if current_mask_index < len(MASKS_DATA):
                        del MASKS_DATA[current_mask_index]
                
                current_mask_index = len(MASKS_DATA)

                SCALED_SAM_POINTS = []
                SAM_LABELS = []
                old_num_points = 0
                IMG_CANVAS = fn.display_saved_masks(MASKS_DATA, img_resized, size_scale)
                cv2.imshow('Image', IMG_CANVAS)
            # Load previous mask (cursor left)
            elif key == 2424832:
                # only if there is data
                if len(MASKS_DATA) > 0:
                    current_mask_index -= 1
                    if current_mask_index < 0:
                        current_mask_index = 0
                    SCALED_SAM_POINTS, SAM_LABELS = fn.load_mask_data(MASKS_DATA, current_mask_index, size_scale)
                    old_num_points = 0
                print(f"Mask Index = {current_mask_index}.")
            # Load next mask (cursor right)
            elif key == 2555904:
                # only if there is data
                if len(MASKS_DATA) > 0:
                    current_mask_index += 1
                    if current_mask_index  >= len(MASKS_DATA):
                        current_mask_index = len(MASKS_DATA) - 1

                    SCALED_SAM_POINTS, SAM_LABELS = fn.load_mask_data(MASKS_DATA, current_mask_index, size_scale)
                    old_num_points = 0
                print(f"Mask Index = {current_mask_index}.")
            # Press "+" to increase image size
            elif key == 43:
                orig_points = [[int(p[0] / size_scale), int(p[1] / size_scale)] for p in SCALED_SAM_POINTS]
                size_scale += 0.25
            # Press "-" to decrease image size
            elif key == 45:
                orig_points = [[int(p[0] / size_scale), int(p[1] / size_scale)] for p in SCALED_SAM_POINTS]
                size_scale -= 0.25
                if size_scale <= 0.25:
                    size_scale = 0.25
            # Resize stuff in case a resizing key was pressed
            if key == 43 or key == 45:
                img_resized = cv2.resize(img_orig, None, fx=size_scale, fy=size_scale)
                IMG_CANVAS = np.copy(img_resized)
                h_canv, w_canv, _ = IMG_CANVAS.shape
                print(f"Vis Image: {h_canv}x{w_canv}")
                IMG_CANVAS = fn.display_all_masks(MASKS_DATA, current_mask_index, mask_u8, IMG_CANVAS, size_scale)
                SCALED_SAM_POINTS = [[int(p[0]*size_scale), int(p[1]*size_scale)] for p in orig_points]
                fn.draw_points(IMG_CANVAS, SCALED_SAM_POINTS, SAM_LABELS, COLOR_OBJ, COLOR_BG)
                cv2.imshow('Image', IMG_CANVAS)
        if exit_flag:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input arguments')
    parser.add_argument('directory', type=str, help='Path to the project directory.')
    parser.add_argument('--fast', action='store_true', help='Enable fast mode.')
    
    args = parser.parse_args()
    project_directory = args.directory
    fast_mode = args.fast

    # Check if model exist
    if fast_mode:
        model_name = "FastSAM"
        checkpoint_name = "FastSAM-x.pt"
        model_type = "YOLOv8x"
        model_download_path = "https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view"
    else:
        model_name = "SAM"
        checkpoint_name = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        model_download_path = "https://dl.fbaipublicfiles.com/segment_anything/"
    
    home = os.getcwd()
    checkpoint_path = os.path.join(home, "models", checkpoint_name)

    if os.path.exists(checkpoint_path):
        print(f"{model_name} found in {checkpoint_path}!")
    else:
        print(f"{model_name} weights NOT FOUND in {checkpoint_path}")
        print(f"Please download {checkpoint_name} from {model_download_path} "
              "and put it inside the models directory")
        sys.exit(0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    # Check if images directory exists
    images_path = os.path.join(project_directory, "images")
    if not os.path.exists(images_path):
        print("The provided path must contain a directory called images.")
        sys.exit(1)

    # Create output masks
    masks_path = os.path.join(project_directory, "masks")
    if not os.path.exists(masks_path):
        os.makedirs(masks_path)

    # Create output embeddings
    emb_path = os.path.join(project_directory, "embeddings")
    if not os.path.exists(emb_path):
        os.makedirs(emb_path)

    IMG = np.zeros((512, 512, 3), np.uint8)
    IMG_CANVAS = np.copy(IMG)
    cv2.imshow('Image', IMG_CANVAS)
    cv2.setMouseCallback('Image', click_event)

    # Masks data is a list. Every list element is a dictionary containing the points, labels, and the associated mask
    MASKS_DATA = []
    #    {"SCALED_SAM_POINTS": [], "sam_labels": [], "mask": None}
    #    for _ in range(20)
    #]

    SCALED_SCALED_SAM_POINTS = []
    SAM_LABELS = []

    main_args = {"project_directory": project_directory,
                 "fast_mode": fast_mode,
                 "checkpoint_name": checkpoint_name,
                 "checkpoint_path": checkpoint_path,
                 "model_type": model_type,
                 "device": device
                }
    main(main_args)
