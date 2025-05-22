from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import numpy as np
import mediapipe as mp
import torch
from PIL import Image
from .models import ResNet50, LSTMPyTorch, pth_processing, get_box
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
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def index(request):
    return render(request, 'face_detection/index.html')

def process_frame(frame):
    # Frame boyutunu küçült
    frame = cv2.resize(frame, (640, 480))
    
    # BGR'den RGB'ye dönüştür
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    
    # Yüz tespiti
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Yüz koordinatlarını al
            idx_to_coors = get_box(face_landmarks, w, h)
            
            # Yüz bölgesini kes
            x_coords = [coords[0] for coords in idx_to_coors.values()]
            y_coords = [coords[1] for coords in idx_to_coors.values()]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Yüz bölgesini genişlet
            padding = 30
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if face_img.size > 0:
                try:
                    # Duygu analizi
                    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    face_tensor = pth_processing(face_pil)
                    face_tensor = face_tensor.to(device)
                    
                    with torch.no_grad():
                        features = resnet.extract_features(face_tensor)
                        features = features.unsqueeze(1)
                        emotion_pred = lstm(features)
                        emotion_probs = torch.softmax(emotion_pred, dim=1)
                        max_prob, emotion_idx = torch.max(emotion_probs, dim=1)
                        
                        # Sadece yüksek güvenilirlikli tahminleri göster
                        if max_prob.item() > 0.5:
                            emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                            emotion = emotions[emotion_idx.item()]
                            
                            # Sonuçları görselleştir
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            cv2.putText(frame, f"{emotion} ({max_prob.item():.2f})", 
                                      (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.9, (0, 255, 0), 2)
                except Exception as e:
                    print(f"İşleme hatası: {str(e)}")
    
    return frame

def video_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("HATA: Kamera açılamadı!")
        return
        
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("HATA: Frame okunamadı!")
            break
            
        # Her 2 frame'de bir işle (performans için)
        if frame_count % 2 == 0:
            processed_frame = process_frame(frame)
        else:
            processed_frame = frame
            
        frame_count += 1
        
        # Frame'i JPEG formatına dönüştür
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_stream(request):
    return StreamingHttpResponse(video_feed(),
                                content_type='multipart/x-mixed-replace; boundary=frame')
