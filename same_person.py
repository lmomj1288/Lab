import cv2
import dlib
import numpy as np

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

db_face_feature_vector = get_face_feature_vector("C:/Users/lmomj/Desktop/AI_Code/age_estimation/img/20_1.jpg")

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    if not ret:
        break

    faces = detector(img, 1)
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        
        # Face recognition
        shape = predictor(img, face)
        face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
        live_face_feature_vector = np.array(face_descriptor)
        distance = np.linalg.norm(db_face_feature_vector - live_face_feature_vector)
        print(f"Euclidean distance: {distance:.3f}")
        
        if distance < 0.4:
            text = "same"
            # 화면에 텍스트 출력
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            text = "new"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 화면에 텍스트 출력
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
    cv2.imshow('Face Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
    
cap.release()
cv2.destroyAllWindows()
