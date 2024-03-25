# Auto-Seg-Annotation

Author: Juan Terven  
Date: August 2023  
License: MIT  
Automatic Segmentation Annotation tool based on SAM (Segment Anything Model)  

## Description
Auto-Seg-Annotation is a tool designed to streamline the process of segmentation annotation. Leveraging the power of the SAM (Segment Anything Model), it provides an intuitive interface and efficient tools for annotating images.

## Installation

1. Clone the repository
```bash
git clone https://github.com/jrterven/auto-seg-annotation.git
```
2. Navigate to the project directory
```bash
cd auto-seg-annotation
```
3. I you have anaconda, create a new conda environment or activate the desired one.
Create new environment:
```bash
conda create --name auto_seg python=3.10
```
Activate existint one:
```bash
conda activate my_env
```

4. Install requirements:
```bash
 pip install -r requirements.txt
```
Then install Pytorch:
with GPU
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
OR without GPU (CPU only)
```bash
pip install torch torchvision torchaudio
```

## Usage
First you need a project directory with a directory called **images** containing the images you want to segment.

Run the tool as:
```bash
 python main <the path to the project directory>
```

In the case of not having a GPU, run the code with the flag --fast to use a lighter model.  
For example:
```bash
 python main <the path to the project directory> --fast
```

The tool will create two additions directories inside the proyect directory:
* **masks**: this directory will contain the resulting segmentation masks.
* **embeddings**: this directory will contain the SAM embeddings so that once the embedding is computed for an image, it will load the embedding the next time instead of creating the embedding.

### Keys functionality 
#### Basic functionality
* **'+'**: Zoom in the image
* **'-'**: Zoom out the image
* **'n'**: Move to next image and save changes
* **'b'**: Move to previous image and save changes
* **enter**: Save current mask information and allow annotate another mask in the same image.
* **'s'**:  Save all the masks, embeddings, and points on disk for the current image.
* **'q'** or **ESC**: exit

#### Other functionality
* **'c'**: Clear all the points of the current mask
* **'d'**: Delete last point
* **'m'**: Show or hide all the masks from visualization
* **'p'**: Show or hide points from visualization
* **CURSOR_LEFT**: Move to the previous object and show its segmentation points (tested only in Windows).
 * **CURSOR_RIGHT**: Move to the next object and show its segmentation points (tested only in Windows).


## Video Tutorials
Tutorial en Español: https://youtu.be/E9hIp2VbSGw