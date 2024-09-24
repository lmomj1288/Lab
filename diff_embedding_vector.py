import numpy as np

# 두 개의 128차원 임베딩 벡터
embedding_vector1 = np.load('C:/Users/lmomj/Desktop/AI_Code/age_estimation/db_face_vectors/db_face_vector_joonjae.npy')
embedding_vector2 = np.load('C:/Users/lmomj/Desktop/AI_Code/age_estimation/db_face_vectors/db_face_vector_LMJ.npy')





print(embedding_vector2)


# # 유클리드 거리 계산
# euclidean_distance = np.linalg.norm(embedding_vector1 - embedding_vector2)

# print(f"두 벡터 간 유클리드 거리: {euclidean_distance}")