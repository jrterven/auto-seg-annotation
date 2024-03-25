import cv2
import matplotlib.pyplot as plt

def format_coord(x, y, img):
    """
    Custom formatter function to show pixel values on hover.
    x, y: coordinates of the mouse cursor.
    img: the image array.
    """
    numcols, numrows = img.shape[:2]
    if x >= 0 and x < numrows and y >= 0 and y < numcols:
        z = img[int(y),int(x)]
        return f'x={x:.0f}, y={y:.0f}, value={z}'
    else:
        return f'x={x:.0f}, y={y:.0f}'

# Load the PNG image
img_path = "seg_sample_project/masks/1001.png" # Update this to your image's path
img = cv2.imread(img_path)

# Convert BGR image to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
fig, ax = plt.subplots()
ax.imshow(img_rgb)

# Customize the formatter to show the pixel values on hover
ax.format_coord = lambda x, y: format_coord(x, y, img_rgb)

plt.show()
