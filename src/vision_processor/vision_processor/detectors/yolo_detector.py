import os
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path=None): 
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '..', 'models', 'best.pt')
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        
        for box in results.boxes:
            confidence = box.conf[0].item()
            if confidence > 0.5:
                x_center, y_center, w, h = box.xywh[0].tolist()
                
                
                return (x_center, y_center)
                
       
        return None
    

    #=============== 검증해봐야
    # --- (기존 코드 유지: class YoloDetector 부분) ---

# 🌟 아래부터 복사해서 yolo_detector.py 맨 밑에 붙여넣으세요!
if __name__ == "__main__":
    import cv2
    import sys

    print("✅ YOLO 단독 테스트 모드를 시작합니다.")
    
    # 1. 디텍터 부품 생성 (best.pt를 자동으로 불러옵니다)
    detector = YoloDetector()

    # 2. 테스트할 mp4 파일 경로 지정 (다니엘 님이 가진 영상 경로로 꼭 바꿔주세요!)
    # 예시: 같은 폴더에 test.mp4가 있다면 파일명만 적어도 됩니다.
    video_path = "test.mp4" 
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ 비디오 파일을 열 수 없습니다: {video_path}")
        print("파일 이름이나 경로가 정확한지 확인해주세요.")
        sys.exit()

    print("▶️ 영상 재생을 시작합니다. (종료하려면 영상 창을 누르고 'q'를 누르세요)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("끝까지 재생되었거나 프레임을 읽을 수 없습니다.")
            break

        # 3. 우리가 만든 디텍터로 바구니 찾기!
        detection = detector.detect(frame)

        # 4. 찾았다면 빨간색 동그라미 그리기
        if detection is not None:
            x, y = detection
            # 이미지(frame)에, (x, y) 좌표를 중심으로, 반지름 15인 빨간색(0, 0, 255) 원을 꽉 채워서(-1) 그립니다.
            cv2.circle(frame, (int(x), int(y)), 15, (0, 0, 255), -1)
            print(f"🎯 타겟 발견! 좌표: ({int(x)}, {int(y)})")
        else:
            print("👀 타겟 없음")

        # 5. 화면에 보여주기
        # 너무 큰 영상은 화면을 벗어날 수 있으니 보기 좋게 사이즈를 살짝 줄여서 출력 (선택 사항)
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("YOLO Detector Test", display_frame)

        # 'q' 키를 누르면 강제 종료 (약 30ms 대기하여 영상 속도 조절)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("사용자에 의해 종료되었습니다.")
            break

    # 메모리 정리
    cap.release()
    cv2.destroyAllWindows()