import os
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path=None): 
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '..', 'models', 'best.pt')
        self.model = YOLO(model_path)
        self.model.to('cpu') #Gpu좋으면 주석 처리
        
    def detect(self, frame):
        #results = self.model(frame, verbose=False)[0]
        results = self.model(frame, imgsz=320, verbose=False)[0] # 컴 좋은면 위에거로
        
        for box in results.boxes:
            confidence = box.conf[0].item()
            if confidence > 0.5:
                x_center, y_center, w, h = box.xywh[0].tolist()
                
                
                return (x_center, y_center)
                
       
        return None
    

    #=============== 검증해봐야
    # --- (기존 코드 유지: class YoloDetector 부분) ---

# 아래부터 복사해서 yolo_detector.py 맨 밑에 붙여넣으세요!
if __name__ == "__main__":
    import cv2
    import sys

    print("YOLO 단독 테스트 모드를 시작합니다.")
    
    # 1. 디텍터 부품 생성 (best.pt를 자동으로 불러옵니다)
    detector = YoloDetector()

    # 2. 테스트할 mp4 파일 경로 지정 
    #current_dir = os.path.dirname(os.path.abspath(__file__))
    #video_path = os.path.join(current_dir, 'videos', 'test.mp4')
    video_path = "/home/daniel/Downloads/test.MP4"
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        print("파일 이름이나 경로가 정확한지 확인해주세요.")
        sys.exit()
    
    start_seconds = 40
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30.0  
        
    start_frame = int(start_seconds * video_fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print(f"영상을 {start_seconds}초 위치({start_frame}번째 프레임)로 건너뛰었습니다.")

    print("영상 재생을 시작합니다. (종료하려면 영상 창을 누르고 'q'를 누르세요)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("끝까지 재생되었거나 프레임을 읽을 수 없습니다.")
            break

        detection = detector.detect(frame)

        if detection is not None:
            x, y = detection
            
            cv2.circle(frame, (int(x), int(y)), 15, (0, 0, 255), -1)
            print(f"타겟 발견! 좌표: ({int(x)}, {int(y)})")
        else:
            print("타겟 없음")

       
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("YOLO Detector Test", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("사용자에 의해 종료되었습니다.")
            break


    cap.release()
    cv2.destroyAllWindows()