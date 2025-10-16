import threading
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Lazy singleton + lock (thread-safe)
_FACE_MESH = None
_LOCK = threading.Lock()

def _get_face_mesh():
    global _FACE_MESH
    if _FACE_MESH is None:
        with _LOCK:
            if _FACE_MESH is None:
                _FACE_MESH = mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=False
                )
    return _FACE_MESH

def get_face_landmarks(image_rgb):
    """
    image_rgb: RGB ndarray (H,W,3)
    return: 468 landmark list or None
    """
    fm = _get_face_mesh()
    with _LOCK:
        results = fm.process(image_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None

def landmark_to_pixel(landmark, image_width, image_height):
    return int(landmark.x * image_width), int(landmark.y * image_height)
