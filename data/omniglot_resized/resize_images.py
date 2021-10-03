from PIL import Image
import glob

image_path = '*/*/'

all_images = glob.glob(image_path + '*')

i = 0

for image_file in all_images:
    im = Image.open(image_file)
    im = im.resize((28,28), resample=Image.LANCZOS)
    im.save(image_file)
    i += 1

    if i % 200 == 0:
        print(i)
