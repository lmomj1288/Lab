import cv2, glob
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

gender_list = ['0','1']

## male : 0, female : 1## 

detector = MTCNN()

model = load_model('C:/Users/lmomj/Desktop/AI_Code/age_estimation/models/gender_predict.h5')

extensions = ['jpg','png','jpeg']

img_list = list(set([item for extension in extensions for item in glob.glob(f'img/*.{extension}')]))

for img_path in img_list:
  img = cv2.imread(img_path)

  faces = detector.detect_faces(img)

  for face in faces:
    x1, y1, width, height = face['box']
    x2, y2 = x1 + width, y1 + height

    face_img = img[y1:y2, x1:x2].copy()
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype("float") / 255.0
    face_img = img_to_array(face_img)
    face_img = np.expand_dims(face_img, axis=0)

    # predict age
    gender_preds = model.predict(face_img)
    gender = gender_list[np.argmax(gender_preds)]

    # visualize
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 2)
    overlay_text = '%s' % (gender)
    cv2.putText(img, overlay_text, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
      fontScale=1, color=(0,0,0), thickness=10)
    cv2.putText(img, overlay_text, org=(x1, y1),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)

  cv2.imshow('img', img)
  cv2.imwrite('result/%s' % img_path.split('/')[-1], img)

  key = cv2.waitKey(0) & 0xFF
  if key == ord('q'):
    break
