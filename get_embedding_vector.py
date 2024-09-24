# db_face_vector.npy : 이명진
# db_face_vector1.npy : 이순재
# db_face_vector2.npy : 외국 여자

import cv2
import dlib
import os 
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
import time

start = time.time()

# dlib 얼굴 탐지기 및 특징 추출 모델 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/lmomj/Desktop/AI_Code/age_estimation/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("C:/Users/lmomj/Desktop/AI_Code/age_estimation/dlib_face_recognition_resnet_model_v1.dat")

# DB 얼굴 특징 벡터
def get_face_feature_vector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        raise ValueError("Cannot find any face")
    face = faces[0]
    shape = predictor(gray, face)
    face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
    return np.array(face_descriptor)

db_face_feature_vector = get_face_feature_vector("C:/Users/lmomj/Desktop/AI_Code/age_estimation/img/LMJ.jpg")
db_face_vector_path = "C:/Users/lmomj/Desktop/AI_Code/age_estimation/db_face_vectors/"
if not os.path.exists(db_face_vector_path):
    os.makedirs(db_face_vector_path)
db_face_vector_file = os.path.join(db_face_vector_path, "db_face_vector_LMJ.npy")
np.save(db_face_vector_file, db_face_feature_vector)
