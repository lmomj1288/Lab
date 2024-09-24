import ray
import cv2, glob
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Ray 초기화
ray.init()

# 나이 추정 모델과 감정 분류 모델 로드
age_model_path = 'C:/Users/lmomj/Desktop/AI_Code/age_estimation/age_model/model_age_estimation14.h5'
emotion_model_path = 'C:/Users/lmomj/Desktop/AI_Code/age_estimation/estimation_model/model_estimation2.h5'

age_model = load_model(age_model_path)
emotion_model = load_model(emotion_model_path)

# MTCNN 얼굴 감지기
detector = MTCNN()

# 이미지 확장자
extensions = ['jpg', 'png', 'jpeg']
img_list = list(set([item for extension in extensions for item in glob.glob(f'img/*.{extension}')]))

@ray.remote
def process_age(img_path):
    img = cv2.imread(img_path)
    faces = detector.detect_faces(img)
    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        face_img = img[y1:y2, x1:x2].copy()
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (128, 128))
        face_img = face_img.astype("float") / 255.0
        face_img = img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)

        age_preds = age_model.predict(face_img)
        age = '50 under' if np.argmax(age_preds) == 0 else '50 over'
        print(f"{img_path.split('/')[-1]}: Age - {age}")

@ray.remote
def process_emotion(img_path):
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

        emotion_preds = emotion_model.predict(face_img)
        emotion = 'Red' if np.argmax(emotion_preds) == 0 else 'Green'
        print(f"{img_path.split('/')[-1]}: Emotion - {emotion}")

# Ray를 사용하여 이미지 리스트에 대해 각 함수를 병렬로 실행
ray.get([process_age.remote(img_path) for img_path in img_list])
ray.get([process_emotion.remote(img_path) for img_path in img_list])

ray.shutdown()