from processing import img_processing, gif_processing
import cv2

process_images = False
process_gif = True
processing_mode = 'BLUR'


# Cats detector
file_path_prefix = './pictures/'

# Images processing
if process_images:
    image_path_prefix = f'{file_path_prefix}pic_'
    for i in range(1, 3):
        image_path = f"{image_path_prefix}{i}.jpg"
        image = cv2.imread(image_path)
        img_processing(image, i, True, processing_mode)


# Раскадровка видео
# делать плавный переход от размытия (по краям) к четкости (в центре, где котик)
if process_gif:
    gif_path = f'{file_path_prefix}gif_1.gif'
    gif_processing(gif_path, processing_mode)
