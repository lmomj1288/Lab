import cv2
import dlib
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

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
db_face_vector_path = "C:/Users/lmomj/Desktop/AI_Code/age_estimation/db_face_vectors/"
db_face_feature_vectors = []
db_face_vector_files = []
for file_name in os.listdir(db_face_vector_path):
    if file_name.endswith(".npy"):
        file_path = os.path.join(db_face_vector_path, file_name)
        db_face_feature_vectors.append(np.load(file_path))
        db_face_vector_files.append(file_name)

# 이미지 파일 경로 입력
image_path = "C:/Users/lmomj/Desktop/image.jpg"  # 원하는 이미지 파일 경로로 변경

# 이미지 읽기
img = cv2.imread(image_path)

faces = detector(img, 1)

if len(faces) >= 1:  # 얼굴이 하나 이상 탐지되었을 때 처리
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        # 얼굴 영역의 크기와 위치를 확인하여 적절한 프레임인지 판단
        face_area = (x2 - x1) * (y2 - y1)
        frame_area = img.shape[0] * img.shape[1]
        face_ratio = face_area / frame_area

        if face_ratio > 0.1:  # 얼굴 영역이 프레임의 20% 이상일 때만 처리
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
            print("Age :", age)
            print("Emotion :", emotion)

            # Face recognition
            shape = predictor(img, face)
            face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
            live_face_feature_vector = np.array(face_descriptor)

            distances = np.linalg.norm(np.array(db_face_feature_vectors) - live_face_feature_vector, axis=1)
            min_distances = np.where(distances < 0.5)[0]

            if len(min_distances) > 0:
                matched_files = [db_face_vector_files[idx] for idx in min_distances]
                print("기존 회원:")
                for file_name in matched_files:
                    print("기존 회원:", file_name.split('.')[0])
            else:
                print("등록되지 않은 회원입니다.")

    # 결과 이미지 출력
    cv2.imshow('Age and Emotion Estimation', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("이미지에서 얼굴을 찾을 수 없습니다.")