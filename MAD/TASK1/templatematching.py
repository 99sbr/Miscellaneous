import time
import json
import os, glob
import cv2
import numpy as np
import imutils
from tqdm import tqdm

start = time.time()
final_result = {}
for imagePath in tqdm(glob.glob(r"/Users/subir/Codes/Miscellaneous/MAD/TASK1/sample_testset/images" + "/*.jpg")):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop_match_list = []
    for templatePath in glob.glob("/Users/subir/Codes/Miscellaneous/MAD/TASK1/sample_testset/crops" + "/*.jpg"):
        template = cv2.imread(templatePath, 0)
        (tH, tW) = template.shape[:2]
        found = None
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            position = np.where(result >= 0.95)
            for point in zip(*position[::-1]):
                found = (point, r)
        if found:
            (maxLoc, r) = found
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
            crop_match_list.append([os.path.basename(templatePath).replace('.jpg', ''), [startX, startY, endX, endY]])

    final_result[os.path.basename(imagePath).replace('.jpg', '')]=[]
    final_result[os.path.basename(imagePath).replace('.jpg', '')] = crop_match_list

with open('data2.json', 'w', encoding='utf-8') as f:
    json.dump(final_result, f, ensure_ascii=False, indent=4)

# print(final_result)
end = time.time()
print("--- %s seconds Execution Time---" % (end - start))
