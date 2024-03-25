import tkinter as tk
import sys
from tkinter import Canvas, font, filedialog, ttk, Listbox, messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import cv2
import numpy as np
import torch
import json
from utils import functions as fn
from constants import COLOR_OBJ, COLOR_BG

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
except:
    print("SAM is not installed. Please install with:")
    print("pip install git+https://github.com/facebookresearch/segment-anything.git")
    sys.exit(1)


def canvas_click(event):
    global SAM_POINTS, SAM_VALUES, IMG_CANVAS

    canvas = event.widget

    # Check if left mouse button was clicked
    x, y = event.x, event.y
    for i, (px, py) in enumerate(SAM_POINTS):
        distance = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
        if distance <= 3:
            del SAM_POINTS[i]
            del SAM_VALUES[i]
            update_status_text(f"Removed point at ({px}, {py})", color="red")
            break
    else:
        SAM_POINTS.append([x, y])
        SAM_VALUES.append(1)

    update_status_text(f"{len(SAM_POINTS)} points:", color="black")

    compute_and_show_sam_mask(canvas, SAM_POINTS, SAM_VALUES)
    canvas.focus_set() # The canvas needs focus to receive key events



def canvas_right_click(event):
    global SAM_POINTS, SAM_VALUES, IMG_CANVAS

    canvas = event.widget

    # Check if right mouse button was clicked
    x, y = event.x, event.y
    # Again, adapt drawing functionality to PIL and Tkinter
    # update_canvas_with_drawing(canvas, IMG_CANVAS, 
    #                            x, y, COLOR_BG)
    SAM_POINTS.append([x, y])
    SAM_VALUES.append(0)
    update_status_text(f"{len(SAM_POINTS)} points:", color="black")

    compute_and_show_sam_mask(canvas, SAM_POINTS, SAM_VALUES)
    canvas.focus_set() # The canvas needs focus to receive key events


def canvas_enter(event):
    save_data()
    

def save_data():
    global MASKS_DATA, CANVAS, SAM_POINTS, SAM_VALUES
    global IMG_CANVAS, CURRENT_MASK_INDEX, CLEAN_IMG

    if len(SAM_POINTS) > 0:
        cat_id, cat_name = get_selected_category_name_and_id()

        masks_dict = {"sam_points": SAM_POINTS, "sam_labels": SAM_VALUES,
                      "mask": MASK_U8, "category_id": cat_id,
                      "category_name": cat_name}
        
        if CURRENT_MASK_INDEX >= len(MASKS_DATA):
            MASKS_DATA.append(masks_dict)
        else:
            MASKS_DATA[CURRENT_MASK_INDEX] = masks_dict
    # In case no points, 
    # e.g. the user pressed enter with no points
    # e.g. the user deleted all the points and press enter
    else:
        if CURRENT_MASK_INDEX < len(MASKS_DATA):
            del MASKS_DATA[CURRENT_MASK_INDEX]
    
    CURRENT_MASK_INDEX = len(MASKS_DATA)

    # restart the points and values
    SAM_POINTS = []
    SAM_VALUES = []

    IMG_CANVAS = fn.display_saved_masks(MASKS_DATA, CLEAN_IMG, 1.0)
    update_canvas_from_cv(CANVAS, IMG_CANVAS)

    # Save the masks and points
    update_status_text("Saving data", color="black")

    fn.save_masks(MASKS_DATA, MASKS_PATH, IMAGE_NAME)
    fn.save_points_and_labels(MASKS_DATA, EMB_PATH, IMAGE_NAME)

    fn.save_coco_annotation(ANNOTATIONS, IMAGE_NAME, MASKS_DATA, OUT_COCO_PATH)


def clear_points():
    global SAM_POINTS, SAM_VALUES, CANVAS

    update_status_text("Clearing all points from current object.", color="black")
    SAM_POINTS = []
    SAM_VALUES = []
    compute_and_show_sam_mask(CANVAS, SAM_POINTS, SAM_VALUES)
    save_data()


def delete_annotations():
    global MASKS_DATA, SAM_POINTS, SAM_VALUES, CANVAS

    update_status_text("Deleting all annotations from current image", color="black")
    SAM_POINTS = []
    SAM_VALUES = []
    MASKS_DATA = []
    save_data()


def compute_and_show_sam_mask(canvas, sam_points, sam_values):
    global PREDICTOR, IMG_CANVAS, MASKS_DATA, CLEAN_IMG, MASK_U8

    if sam_points:
        MASK_U8 = fn.predict_masks(PREDICTOR, sam_points,
                               np.array(sam_values), 1.0, False)
    else:
        h_orig, w_orig, _ = CLEAN_IMG.shape
        MASK_U8 = np.zeros((h_orig, w_orig), dtype=np.uint8)

    current_mask_index = len(MASKS_DATA)
    IMG_CANVAS = fn.display_all_masks(MASKS_DATA, current_mask_index, MASK_U8,
                                      CLEAN_IMG, 1.0)
    fn.draw_points(IMG_CANVAS, sam_points, sam_values, COLOR_OBJ, COLOR_BG)
    update_canvas_from_cv(canvas, IMG_CANVAS)


def select_previous_object():
    global MASKS_DATA, SAM_POINTS, SAM_VALUES, CANVAS, CURRENT_MASK_INDEX

    if len(MASKS_DATA) > 0:
        CURRENT_MASK_INDEX -= 1
        if CURRENT_MASK_INDEX < 0:
            CURRENT_MASK_INDEX = 0
        SAM_POINTS, SAM_VALUES = fn.load_mask_data(MASKS_DATA,
                                                   CURRENT_MASK_INDEX, 1.0)
        compute_and_show_sam_mask(CANVAS, SAM_POINTS, SAM_VALUES)
    CANVAS.focus_set() # The canvas needs focus to receive key events
    update_status_text(f"Item #{CURRENT_MASK_INDEX}," + 
                       f"Category {MASKS_DATA[CURRENT_MASK_INDEX]['category_id']}:" +
                       f"{MASKS_DATA[CURRENT_MASK_INDEX]['category_name']}",
                       color="black")


def select_next_object():
    global MASKS_DATA, SAM_POINTS, SAM_VALUES, CANVAS, CURRENT_MASK_INDEX

    if len(MASKS_DATA) > 0:
        CURRENT_MASK_INDEX += 1
        if CURRENT_MASK_INDEX  >= len(MASKS_DATA):
            CURRENT_MASK_INDEX = len(MASKS_DATA) - 1

        SAM_POINTS, SAM_VALUES = fn.load_mask_data(MASKS_DATA,
                                                          CURRENT_MASK_INDEX, 1.0)
        compute_and_show_sam_mask(CANVAS, SAM_POINTS, SAM_VALUES)
    CANVAS.focus_set() # The canvas needs focus to receive key events
    update_status_text(f"Item #{CURRENT_MASK_INDEX}," + 
                       f"Category {MASKS_DATA[CURRENT_MASK_INDEX]['category_id']}:" +
                       f"{MASKS_DATA[CURRENT_MASK_INDEX]['category_name']}",
                       color="black")


# Function to update the canvas, adapting OpenCV drawing to PIL/Tkinter
def update_canvas_with_drawing(canvas, cv_img, x, y, color):
    # Convert the OpenCV BGR color format to RGB for PIL
    color = (color[2], color[1], color[0])  # Assuming color is in BGR format
    cv2.circle(cv_img, (x, y), 3, COLOR_OBJ, -1)
    update_canvas_from_cv(canvas, cv_img)  # You should define this function as shown before


def select_project_dir():
    global IMAGES_PATH, MASKS_PATH, EMB_PATH, ANNOTATIONS
    global CATEGORIES_LIST, OUT_COCO_PATH
    
    # Show project directory
    project_path = filedialog.askdirectory()

    # if project directory exists
    if project_path:
        images_path = os.path.join(project_path, "images")
        masks_path = os.path.join(project_path, "masks")
        emb_path = os.path.join(project_path, "embeddings")

        # it must have a directory called images
        if os.path.exists(images_path):

            # create the masks and embeddings directories if not existed
            os.makedirs(masks_path, exist_ok=True)
            os.makedirs(emb_path, exist_ok=True)

            # set the vars with the paths
            project_path_var.set(project_path)
            IMAGES_PATH = images_path
            MASKS_PATH = masks_path
            EMB_PATH = emb_path
            update_file_list(images_path)

            # Create empty annotations file or load it if existent
            OUT_COCO_PATH = os.path.join(project_path, "annotations.json")
            if not os.path.exists(OUT_COCO_PATH):
                ANNOTATIONS = fn.create_empty_coco(OUT_COCO_PATH)
            else:
                update_status_text("Found 'annotations.json'. Loading the annotations.",
                                   color="blue")
                ANNOTATIONS = fn.load_annotations(OUT_COCO_PATH)
                CATEGORIES_LIST = ANNOTATIONS["categories"]
                updating_categories_box(CATEGORIES_LIST)
        else:
            update_status_text("ERROR: the project path must have a directory called images\n", color="red")


def update_file_list(path):
    file_listbox.delete(0, tk.END)  # Clear the listbox first
    
    # Filter and list image files only
    for filename in os.listdir(path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_listbox.insert(tk.END, filename)
    
    # Automatically select the first file and trigger the display
    if file_listbox.size() > 0:
        file_listbox.select_set(0)  # This will select the first item
        file_listbox.event_generate("<<ListboxSelect>>")


def on_file_select(event):
    global MASKS_DATA, PREDICTOR, DEVICE, IMG_CANVAS, CLEAN_IMG
    global CANVAS, IMAGE_NAME, IMAGES_PATH, EMB_PATH
    global SAM_POINTS, SAM_VALUES, CURRENT_MASK_INDEX

    # Get selected file index
    try:
        reset_item_colors()  # Reset colors for all items
        update_selected_item_colors()  # Apply custom colors to the selected item

        index = file_listbox.curselection()[0]
        IMAGE_NAME = file_listbox.get(index)
        image_path = os.path.join(IMAGES_PATH, IMAGE_NAME)

        # Display image to canvas
        h, w, CLEAN_IMG = load_and_display_image(image_path, CANVAS)
        IMG_CANVAS = CLEAN_IMG.copy()

        # load embedding for the image
        load_embedding(PREDICTOR, EMB_PATH, IMAGE_NAME,
                        CLEAN_IMG, DEVICE)
        
        # load sam points and sam labels for the image
        SAM_POINTS = []
        SAM_VALUES = []
        MASKS_DATA = fn.load_points_and_labels(EMB_PATH, IMAGE_NAME,
                                                PREDICTOR, False, 1.0)

        # Select the last mask
        CURRENT_MASK_INDEX = len(MASKS_DATA)
        update_status_text(f"\nFound {CURRENT_MASK_INDEX} masks in current image.", color="black")

        IMG_CANVAS = fn.display_saved_masks(MASKS_DATA, CLEAN_IMG, 1.0)
        update_canvas_from_cv(CANVAS, IMG_CANVAS)
    except IndexError as exc:
        print("Ignoring clic out of files list")



def load_and_display_image(image_path, canvas):

    # Load and display the selected image
    pil_image = Image.open(image_path)
    image = ImageTk.PhotoImage(pil_image)

    image_np = np.array(pil_image)

    # Convert RGB to BGR (OpenCV uses BGR)
    cv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Update canvas size to fit image
    canvas.config(width=pil_image.width, height=pil_image.height)
    canvas.create_image(0, 0, anchor=tk.NW, image=image)
    canvas.image = image  # Keep a reference!

    return image_np.shape[0], image_np.shape[1], cv_img


def update_canvas(canvas, pil_img):
    # Update the canvas
    canvas.image = pil_img  # Keep a reference, prevent garbage collection
    canvas.create_image(0, 0, anchor=tk.NW, image=pil_img)


def update_canvas_from_cv(canvas, cv_img):
    # Convert the OpenCV BGR format to RGB format
    cv_img2 = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    # Convert the NumPy array (OpenCV image) to a PIL image
    pil_img = Image.fromarray(cv_img2)
    
    # Convert the PIL image to a PhotoImage
    tk_img = ImageTk.PhotoImage(pil_img)
    
    # Update the canvas
    canvas.image = tk_img  # Keep a reference, prevent garbage collection
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)


def load_embedding(predictor, emb_path, image_name, img_orig, device):
    update_status_text("Loading SAM embedding...", color="black")
    got_embedding = fn.load_embedding(predictor, emb_path, image_name, img_orig, device)
    if not got_embedding:
        print("Computing SAM embedding...")
        predictor.set_image(img_orig)

        # Save the embedding for next time
        embedding = predictor.get_image_embedding()
        fn.save_embedding(embedding, emb_path, image_name)


def show_hide_masks():
    global masks_button, SHOW_MASKS

    SHOW_MASKS = not SHOW_MASKS
    if SHOW_MASKS:
        masks_button.config(text="Hide Masks")
    else:
        masks_button.config(text="Show Masks")


def show_hide_points():
    global points_button, SHOW_POINTS

    SHOW_POINTS = not SHOW_POINTS
    if SHOW_POINTS:
        points_button.config(text="Hide Points")
    else:
        points_button.config(text="Show Points")



def show_warning_to_overwrite_categories():
    # This function will show a warning pop-up with Yes or No options.
    response = messagebox.askyesno("Categories Mismatch!",
                                   "The Categories loaded are different from the categories in the annotations. Do you want to overwrite them?")
    return response


def updating_categories_box(categories_list):
    update_combobox_values(categories_list)

    update_status_text("Loading Categories:", color="black")
    categories_names = get_category_ids_and_names(categories_list)
    update_status_text(", ".join(categories_names), color="black")


def select_classes_file():
    global CATEGORIES_LIST, ANNOTATIONS

    classes_file_path = filedialog.askopenfilename()
    # if file selected
    if classes_file_path:
        classes_file_var.set(classes_file_path)
        # load categories
        new_categories_list = fn.load_categories(classes_file_path)

        # it the new categories are different from the annotation categories
        if new_categories_list != ANNOTATIONS["categories"]:
            response = show_warning_to_overwrite_categories()
            # Use the new categories
            if response == True:
                update_status_text("Updating Categories in Annotations", color="red")
                ANNOTATIONS["categories"] = new_categories_list
                CATEGORIES_LIST = new_categories_list
                updating_categories_box(CATEGORIES_LIST)

                # Writing the json_dict to a file at out_path
                with open(OUT_COCO_PATH, 'w') as outfile:
                    json.dump(ANNOTATIONS, outfile, indent=4)
            else:
                update_status_text("Discarding new Categories", color="red")


def insert_colored_text(text_widget, text, color):
    tag_name = f"tag_{color}"
    text_widget.tag_configure(tag_name, foreground=color)
    text_widget.insert(tk.END, text, (tag_name,))


def update_status_text(text, color="red"):
    # Append a newline character to ensure the next text appears on a new line
    formatted_text = text + "\n"
    insert_colored_text(status_text, formatted_text, color)
    status_text.see(tk.END)  # Scroll to the end of the text


# Function to extract names for the Combobox
def get_category_ids_and_names(categories):
    return [f"{category['id']}:{category['name']}" for category in categories]


# Function to extract names for the Combobox
def get_category_names(categories):
    return [category["name"] for category in categories]


def update_combobox_values(new_categories_list):
    # Update the combobox's 'values' property with new category names
    category_combobox['values'] = get_category_ids_and_names(new_categories_list)
    
    # Optionally, set the combobox to show the first item from the new list
    category_combobox.set(f"{new_categories_list[0]['id']}:{new_categories_list[0]['name']}")


def get_selected_category_name_and_id():
    class_id, name = category_combobox.get().split(":")
    return int(class_id), name


def on_category_combobox_select(event):
    global CANVAS

    # This function is called whenever the user selects a value from the combobox.
    class_id, class_name = get_selected_category_name_and_id()
    #print(class_id, class_name)
    CANVAS.focus_set() # The canvas needs focus to receive key events
    

def export_to_coco():
    global MASKS_DATA, OUT_COCO_PATH, IMAGE_NAME






def update_selected_item_colors():
    global file_listbox
    # Apply custom highlight to the currently selected item
    for i in file_listbox.curselection():
        file_listbox.itemconfig(i, {'bg': 'blue', 'fg': 'white'})


def reset_item_colors():
    global file_listbox
    # Reset the colors for all items in the listbox
    for i in range(file_listbox.size()):
        file_listbox.itemconfig(i, {'bg': 'white', 'fg': 'black'})


def on_file_focus_out(event):
    update_selected_item_colors()  # Ensure the selected item retains custom colors when losing focus


def on_file_focus_in(event):
    update_selected_item_colors()  # Re-apply custom colors to the selected item when gaining focus


if __name__ == "__main__":

    model_name = "SAM"
    checkpoint_name = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    model_download_path = "https://dl.fbaipublicfiles.com/segment_anything/"

    # Load model
    home = os.getcwd()
    checkpoint_path = os.path.join(home, "models", checkpoint_name)
    if os.path.exists(checkpoint_path):
        print(f"{model_name} found in {checkpoint_path}!")
    else:
        print(f"{model_name} weights NOT FOUND in {checkpoint_path}")
        print(f"Please download {checkpoint_name} from {model_download_path} "
              "and put it inside the models directory")
        sys.exit(0)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {DEVICE}")

    # Masks data is a list. Every list element is a dictionary 
    # containing the points, labels, and the associated mask
    MASKS_DATA = []
    SAM_POINTS = []
    SAM_VALUES = []
    ANNOTATIONS = {}

    CATEGORIES_LIST = [
        {
            "id": 0,
            "name": "class 1",
            "supercategory": "my classes"
        },
        {
            "id": 1,
            "name": "class 2",
            "supercategory": "my classes"
        },
    ]


    SHOW_MASKS = True
    SHOW_POINTS = True

    CLEAN_IMG = np.zeros((512, 512, 3), np.uint8)
    IMG_CANVAS = CLEAN_IMG.copy()
    MASK_U8 = np.zeros((512, 512), np.uint8)
    IMAGE_NAME = ""
    IMAGES_PATH = ""
    MASKS_PATH = ""
    EMB_PATH = ""
    ANN_PATH = ""
    OUT_COCO_PATH = ""
    CURRENT_MASK_INDEX = 0
    LISTBOX_SELECTED_INDEX = ()

    # Loading SAM model
    print("Loading SAM ...")
    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device=DEVICE)
        PREDICTOR = SamPredictor(sam)
    except Exception as e:
        print(f"Error {e}")
    print("SAM model successfully loaded!")

    root = tk.Tk()
    root.title("Image Automatic Segmentation")
    root.geometry("1300x800")

    custom_font = font.Font(family="Helvetica", size=12)
    project_path_var = tk.StringVar(root)
    classes_file_var = tk.StringVar(root)
    category_id_var = tk.StringVar(root)
    category_id_var.set("0")
    selected_category_name = tk.StringVar(root)

    file_browse_frame = tk.Frame(root)
    file_browse_frame.grid(row=0, column=0, columnspan=4, pady=5, sticky=tk.W)

    # Images Directory Widgets
    file_label = tk.Label(file_browse_frame, text="Images Directory:", font=custom_font)
    file_label.grid(row=0, column=0, padx=5, sticky=tk.W)
    file_entry = tk.Entry(file_browse_frame, textvariable=project_path_var, width=50, font=custom_font)
    file_entry.grid(row=0, column=1, padx=5, sticky=tk.W)
    browse_button = tk.Button(file_browse_frame, text="Browse", command=select_project_dir, font=custom_font)
    browse_button.grid(row=0, column=2, padx=5, sticky=tk.W)

    # Classes File Widgets
    classes_label = tk.Label(file_browse_frame, text="Categories File:", font=custom_font)
    classes_label.grid(row=1, column=0, padx=5, sticky=tk.W)
    classes_entry = tk.Entry(file_browse_frame, textvariable=classes_file_var, width=50, font=custom_font)
    classes_entry.grid(row=1, column=1, padx=5, sticky=tk.W)
    classes_browse_button = tk.Button(file_browse_frame, text="Browse", command=select_classes_file, font=custom_font)
    classes_browse_button.grid(row=1, column=2, padx=5, sticky=tk.W)

    # Frame for Listbox and Canvas
    content_frame = tk.Frame(root)
    content_frame.grid(row=2, column=0, sticky="nsew", pady=5)

    root.grid_rowconfigure(2, weight=1)
    root.grid_columnconfigure(0, weight=1)
    content_frame.grid_rowconfigure(0, weight=1)
    content_frame.grid_columnconfigure(1, weight=1)

    file_listbox = Listbox(content_frame, font=custom_font, selectmode=tk.SINGLE)
    file_listbox.grid(row=0, column=0, sticky="ns")
    file_listbox.bind('<<ListboxSelect>>', on_file_select)
    file_listbox.bind('<FocusIn>', on_file_focus_in)
    file_listbox.bind('<FocusOut>', on_file_focus_out)

    CANVAS = Canvas(content_frame, bg="grey")
    CANVAS = Canvas(content_frame, bg="grey")
    CANVAS.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
    CANVAS.bind("<Button-1>", canvas_click)  # Left click
    CANVAS.bind("<Button-3>", canvas_right_click)  # Right click
    CANVAS.bind('<Return>', canvas_enter)

    right_frame = tk.Frame(content_frame)
    right_frame.grid(row=0, column=2, sticky="ns", padx=5)

    # Sub-frame for class label and entry side by side
    class_sub_frame = tk.Frame(right_frame)
    class_sub_frame.pack(pady=(10, 20))  # Add padding above and below class_sub_frame

    class_label = tk.Label(class_sub_frame, text="Class:", font=custom_font)
    class_label.pack(side=tk.LEFT, padx=(0, 5))

    category_combobox = ttk.Combobox(class_sub_frame, textvariable=selected_category_name,
                                    values=get_category_ids_and_names(CATEGORIES_LIST),
                                    font=custom_font)
    category_combobox.bind("<<ComboboxSelected>>", on_category_combobox_select)
    category_combobox.pack(side=tk.LEFT)

    # Optionally set a default value
    category_combobox.set(CATEGORIES_LIST[0]["name"])  # Set default to first category name

    # Sub-frame for Previous and Next object buttons side by side
    prev_next_sub_frame = tk.Frame(right_frame)
    prev_next_sub_frame.pack(pady=(0, 10))  # Add padding below prev_next_sub_frame


    prev_obj_button = tk.Button(prev_next_sub_frame, text="Prev Object",
                             command=select_previous_object, font=custom_font, anchor='w')
    prev_obj_button.pack(side=tk.LEFT, padx=(0, 5))
    
    next_obj_button = tk.Button(prev_next_sub_frame, text="Next Object",
                              command=select_next_object, font=custom_font, anchor='w')
    next_obj_button.pack(side=tk.LEFT)


    # Sub-frame for masks and points buttons side by side
    masks_and_points_sub_frame = tk.Frame(right_frame)
    masks_and_points_sub_frame.pack()

    masks_button = tk.Button(masks_and_points_sub_frame, text="Hide masks",
                             command=show_hide_masks, font=custom_font, anchor='w')
    masks_button.pack(side=tk.LEFT, padx=(0, 5))
    
    points_button = tk.Button(masks_and_points_sub_frame, text="Hide points",
                              command=show_hide_points, font=custom_font, anchor='w')
    points_button.pack(side=tk.LEFT)

    # Sub-frame for clear and delete buttons side by side
    clear_and_delete_sub_frame = tk.Frame(right_frame)
    clear_and_delete_sub_frame.pack()

    # Clear points button
    clear_points_button = tk.Button(clear_and_delete_sub_frame, text="Clear Points",
                                    command=clear_points, font=custom_font, anchor='w')
    clear_points_button.pack(side=tk.LEFT, padx=(0, 5))

    # Delete all annotations from image
    delete_annotations_button = tk.Button(clear_and_delete_sub_frame, text="Delete All", command=delete_annotations,
                            font=custom_font, anchor='w')
    delete_annotations_button.pack(side=tk.LEFT)

    # Save button
    save_button = tk.Button(right_frame, text="Save Annotation", command=save_data,
                            font=custom_font, anchor='w')
    save_button.pack(pady=5, anchor='w')

    # Export button
    export_coco_button = tk.Button(right_frame, text="Export to COCO", command=export_to_coco,
                            font=custom_font, anchor='w')
    export_coco_button.pack(pady=5, anchor='w')

    # Create a sub-frame in the right_frame for the status text and scrollbar
    status_frame = tk.Frame(right_frame)
    status_frame.pack(fill=tk.BOTH, expand=True)

    # Status text
    status_text = tk.Text(status_frame, wrap=tk.WORD, height=5, width=20, font=custom_font)  # Adjust width as needed
    status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Status scrollbar
    status_scrollbar = tk.Scrollbar(status_frame, command=status_text.yview)
    status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    status_text.config(yscrollcommand=status_scrollbar.set)


    # Additional space for more components can be added here
    # For example, you could add more buttons or entries in a similar manner to the save_button and class_entry

    root.mainloop()

    