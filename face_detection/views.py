from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
import cv2
import numpy as np
import mediapipe as mp
import torch
from PIL import Image
from .models import ResNet50, LSTMPyTorch, pth_processing, get_box
from .scoring_service import ScoringService
import os

# Model yükleme
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = ResNet50(num_classes=7, channels=3)
lstm = LSTMPyTorch()

# Model ağırlıklarının yolları
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
resnet_weights_path = os.path.join(BASE_DIR, 'face_detection', 'models', 'FER_static_ResNet50_AffectNet.pt')
lstm_weights_path = os.path.join(BASE_DIR, 'face_detection', 'models', 'FER_dinamic_LSTM_Aff-Wild2.pt')

# Model ağırlıklarını yükle
resnet.load_state_dict(torch.load(resnet_weights_path, map_location=device))
lstm.load_state_dict(torch.load(lstm_weights_path, map_location=device))

resnet.to(device)
lstm.to(device)
resnet.eval()
lstm.eval()

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Emotion dictionary matching run_webcam.ipynb
DICT_EMO = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}

# Initialize ScoringService
scoring_service = ScoringService()

def index(request):
    return render(request, 'face_detection/index.html')

def process_frame(frame):
    # BGR'den RGB'ye dönüştür
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    
    # Yüz tespiti
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Yüz koordinatlarını al
            idx_to_coors = get_box(face_landmarks, w, h)
            
            # Get coordinates from dictionary
            x_coords = [coords[0] for coords in idx_to_coors.values()]
            y_coords = [coords[1] for coords in idx_to_coors.values()]
            
            startX = min(x_coords)
            startY = min(y_coords)
            endX = max(x_coords)
            endY = max(y_coords)
            
            # Yüz bölgesini kes
            face_img = frame[startY:endY, startX:endX]
            
            if face_img.size > 0:
                # Ensure minimum face size
                if face_img.shape[0] < 48 or face_img.shape[1] < 48:
                    continue
                    
                # Duygu analizi
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                face_tensor = pth_processing(face_pil)
                face_tensor = face_tensor.to(device)
                
                with torch.no_grad():
                    features = torch.nn.functional.relu(resnet.extract_features(face_tensor)).detach().cpu().numpy()
                    
                    # Initialize or update LSTM features
                    if not hasattr(process_frame, 'lstm_features'):
                        process_frame.lstm_features = [features] * 10
                    else:
                        process_frame.lstm_features = process_frame.lstm_features[1:] + [features]
                    
                    # Prepare LSTM input
                    lstm_input = torch.from_numpy(np.vstack(process_frame.lstm_features))
                    lstm_input = torch.unsqueeze(lstm_input, 0).to(device)
                    
                    # Get emotion prediction
                    output = lstm(lstm_input).detach().cpu().numpy()
                    emotion_idx = np.argmax(output)
                    confidence = output[0][emotion_idx]
                
                # Get emotion label
                emotion = DICT_EMO[emotion_idx]
                
                # Record emotion in scoring service
                scoring_service.record(emotion, confidence)
                
                # Sonuçları görselleştir
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                
                # Display emotion and confidence
                text = f"{emotion} ({confidence:.1%})"
                cv2.putText(frame, text, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

def video_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Frame'i işle
        processed_frame = process_frame(frame)
        
        # Frame'i JPEG formatına dönüştür
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_stream(request):
    return StreamingHttpResponse(video_feed(),
                                content_type='multipart/x-mixed-replace; boundary=frame')

def get_questions(request):
    """API endpoint to get current questions"""
    questions = scoring_service.get_questions()
    return JsonResponse({'questions': questions})
