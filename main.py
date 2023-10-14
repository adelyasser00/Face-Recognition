import numpy as np
import os
from PIL import Image

# lists for storing the data matrix D and label vector y
D = []
y = []

# 2)
# loop over all the directory archive
for subject in range(1, 40):
    # every subject has 10 images, get 10 images per subject
    imageCount = 0

    for image in os.listdir(f'archive/s{subject}'):
        if imageCount == 10:
            break
        imageCount += 1
        temp = Image.open(f'archive/s{subject}/{image}')
        vector = np.array(temp).flatten()

        D.append(vector)
        y.append(subject)

# convert the dataMatrix and labels to numpy arrays
D = np.array(D)
y = np.array(y)

# 3) Split the Dataset into Training and Test sets
