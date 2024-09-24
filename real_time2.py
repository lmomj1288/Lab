import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

# 나이와 감정의 범위 설정
age_list = ['60 under', '60 over']
emotion_list = ['Red', 'Green']

# dlib 얼굴 탐지기 및 특징 추출 모델 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/lmomj/Desktop/AI_Code/age_estimation/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("C:/Users/lmomj/Desktop/AI_Code/age_estimation/dlib_face_recognition_resnet_model_v1.dat")

# 학습된 모델 불러오기
age_model = load_model('C:/Users/lmomj/Desktop/AI_Code/age_estimation/age_model/model_age_estimation15.h5')
emotion_model = load_model('C:/Users/lmomj/Desktop/AI_Code/age_estimation/estimation_model/model_estimation6.h5')

# DB 얼굴 특징 벡터
def get_face_feature_vector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) ==0:
        raise ValueError("Cannot find any face")
    face = faces[0]
    shape = predictor(gray, face)
    face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
    return np.array(face_descriptor)

db_face_feature_vector = get_face_feature_vector("C:/Users/lmomj/Desktop/AI_Code/age_estimation/img/LMJ.jpg")

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    if not ret:
        break

    faces = detector(img, 1)
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = img[y1:y2, x1:x2].copy()
        face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Age estimation preprocessing
        face_img_age = cv2.resize(face_img_gray, (128, 128))
        face_img_age = face_img_age.astype("float") / 255.0
        face_img_age = img_to_array(face_img_age)
        face_img_age = np.expand_dims(face_img_age, axis=0)

        # Emotion estimation preprocessing
        face_img_emotion = cv2.resize(face_img_gray, (48, 48))
        face_img_emotion = face_img_emotion.astype("float") / 255.0
        face_img_emotion = img_to_array(face_img_emotion)
        face_img_emotion = np.expand_dims(face_img_emotion, axis=0)

        # Prediction
        age_preds = age_model.predict(face_img_age)
        age = age_list[np.argmax(age_preds)]
        emotion_preds = emotion_model.predict(face_img_emotion)
        emotion = emotion_list[np.argmax(emotion_preds)]

        # Visualize result
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        overlay_text = '%s, %s' % (age, emotion)
        cv2.putText(img, overlay_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Face recognition
        shape = predictor(img, face)
        face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
        live_face_feature_vector = np.array(face_descriptor)
        distance = np.linalg.norm(db_face_feature_vector - live_face_feature_vector)
        print(f"Euclidean distance: {distance:.3f}")
        if distance < 0.5:
            print("동일 인물입니다.")
        else:
            print("다른 인물입니다.")
            
    cv2.imshow('Age and Emotion Estimation', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
    
cap.release()
cv2.destroyAllWindows() 
