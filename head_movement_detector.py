import cv2
import mediapipe as mp
import numpy as np

ind2text = {
    0:'left',
    3:'up',
    2:'right',
    1:'down'
    }

def deg2dir(deg):
    if (-22.5 < deg <= 22.5):
        return 0
    elif (22.5 < deg < 112.5):
        return 1
    elif (112.5 <= np.abs(deg) <= 180):
        return 2
    elif (-112.5 < deg <= -22.5):
        return 3
    else:
        print(deg)

def cart2polar(arr):
    polar = np.zeros(arr.shape)
    polar[:,0] = np.sqrt(arr[:,0]**2 + arr[:,1]**2)
    polar[:,1] = np.arctan2(arr[:,1],arr[:,0]) * 180 / np.pi
    return polar

def genhist(arr):

    hist = np.zeros(4)

    for i in range(arr.shape[0]):

        if arr[i][0] < 0.01:
            continue

        hist[deg2dir(arr[i][1])]+=1

    hist /= arr.shape[0]

    return hist

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=0.5, circle_radius=1)
cap = cv2.VideoCapture(1)

text = ''

# r = 0.18533697614631156
# f_d = 0.4210272218051749

landmarks = [i for i in range(478)]

font                   = cv2.FONT_HERSHEY_SIMPLEX

bottomLeftCornerOfText = (400,400)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2

cur_values = np.zeros((len(landmarks),2))

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:

         prev_values = cur_values
         cur_values = np.array([[item.x,item.y] for item in results.multi_face_landmarks[0].landmark])
            
         diff = cur_values - prev_values

         diff_polar = cart2polar(diff)
         
         if diff[:,0].max() < 0.01:
             text = ''
         else:
             diff_polar_hist = genhist(diff_polar)
             
             text = ind2text[np.argmax(diff_polar_hist)]

         for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
    # Flip the image horizontally for a selfie-view display.


    image = cv2.flip(image, 1)
    cv2.putText(image,text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)    

    cv2.imshow('MediaPipe Face Mesh', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

