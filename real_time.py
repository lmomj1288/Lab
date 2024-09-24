import cv2
import dlib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import time 

# 나이와 감정의 범위 설정
age_list = ['60 under', '60 over']
emotion_list = ['Red', 'Green']

# dlib 얼굴 탐지기 초기화
detector = dlib.get_frontal_face_detector()

# 학습된 모델 불러오기
age_model = load_model('C:/Users/lmomj/Desktop/AI_Code/age_estimation/age_model/model_age_estimation15.h5')
emotion_model = load_model('C:/Users/lmomj/Desktop/AI_Code/age_estimation/estimation_model/model_estimation6.h5')
start_time = time.time()
# 카메라 시작
cap = cv2.VideoCapture(0)
print(f"카메라가 켜지는 데 걸린 시간: {time.time() - start_time:.2f}초")
while True:
    # 카메라에서 프레임 읽기
    ret, img = cap.read()

    if not ret:
        break

    # dlib을 사용하여 얼굴 탐지
    faces = detector(img, 1)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        # 얼굴 부분 추출 및 전처리
        face_img = img[y1:y2, x1:x2].copy()
        face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # 나이 예측을 위한 전처리
        face_img_age = cv2.resize(face_img_gray, (128, 128))
        face_img_age = face_img_age.astype("float") / 255.0
        face_img_age = img_to_array(face_img_age)
        face_img_age = np.expand_dims(face_img_age, axis=0)

        # 감정 예측을 위한 전처리
        face_img_emotion = cv2.resize(face_img_gray, (48, 48))
        face_img_emotion = face_img_emotion.astype("float") / 255.0
        face_img_emotion = img_to_array(face_img_emotion)
        face_img_emotion = np.expand_dims(face_img_emotion, axis=0)

        # 나이와 감정 예측
        age_preds = age_model.predict(face_img_age)
        age = age_list[np.argmax(age_preds)]
        emotion_preds = emotion_model.predict(face_img_emotion)
        emotion = emotion_list[np.argmax(emotion_preds)]
        
        # 결과 시각화
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        overlay_text = '%s, %s' % (age, emotion)
        cv2.putText(img, overlay_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 화면에 결과 표시
    cv2.imshow('Age and Emotion Estimation', img)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
