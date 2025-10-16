import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def get_face_landmarks(image_rgb):
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
    return None

def landmark_to_pixel(landmark, image_width, image_height):
    return int(landmark.x * image_width), int(landmark.y * image_height)
