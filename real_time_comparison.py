import cv2
import dlib
import numpy as np

predictor = dlib.shape_predictor("C:/Users/lmomj/Desktop/AI_Code/age_estimation/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("C:/Users/lmomj/Desktop/AI_Code/age_estimation/dlib_face_recognition_resnet_model_v1.dat")

def get_face_feature_vector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    if len(faces) ==0:
        raise ValueError("얼굴을 찾을 수 없음")
    
    face = faces[0]
    shape = predictor(gray,face)
    face_descriptor = face_rec_model.compute_face_descriptor(img,shape)
    
    return np.array(face_descriptor)

cap = cv2.VideoCapture(0)  # 카메라 초기화
detector = dlib.get_frontal_face_detector()
db_face_feature_vector = get_face_feature_vector("C:/Users/lmomj/Desktop/AI_Code/age_estimation/img/LMJ.jpg")  # DB 얼굴 특징 벡터

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        shape = predictor(gray, face)
        face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
        live_face_feature_vector = np.array(face_descriptor)
        
        distance = np.linalg.norm(db_face_feature_vector - live_face_feature_vector)  # 두 벡터 간 거리 계산
        print(f"유클리드 거리: {distance:.3f}")
        if distance < 0.5:  # 거리가 작을수록 얼굴이 유사함을 의미
            # 동일 인물로 판단, 후속 처리
            print("동일 인물입니다.")
        else:
            print("다른 인물입니다.")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
