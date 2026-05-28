import cv2
import numpy as np
import matplotlib.pyplot as plt

num=input("실험 번호 입력: ")
VIDEO_PATH = 'exp'+num+'.mp4'  # 영상 파일 경로 입력

# 2. 비디오 파일 열기
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: 영상 파일을 열 수 없습니다. 경로를 확인하세요: {VIDEO_PATH}")
    exit()

# 영상 스펙 확인
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 분석할 원의 중심을 영상 중앙으로 가정
CENTER_X = width // 2
CENTER_Y = height // 2
MAX_RADIUS = min(width // 2, height // 2)

print(f"[입력 세로 영상 정보] 해상도: {width}x{height} (세로형) | FPS: {fps} | 총 프레임 수: {total_frames}")
print(f"[분석 설정] 원의 중심 ({CENTER_X}, {CENTER_Y})을 기준으로 최대 반경 {MAX_RADIUS}px까지 동심원 분석을 수행합니다.")

# 데이터 저장용 리스트
time_axis = []
radial_profiles_history = [] # 각 프레임의 방사형 프로파일을 저장할 리스트 (2D 배열)

frame_count = 0

# 2.5. 마우스 클릭으로 원의 중심 수동 설정
print("\n[안내] 팝업된 창에서 간섭 무늬의 정중앙을 마우스 왼쪽 버튼으로 클릭하세요.")

# 전역 변수 초기화
CENTER_X, CENTER_Y = width // 2, height // 2
center_selected = False

def get_mouse_click(event, x, y, flags, param):
    global CENTER_X, CENTER_Y, center_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        # 화면을 50% 축소해서 보여주므로, 실제 좌표는 클릭한 위치에 곱하기 2를 해야 함
        CENTER_X = x * 2
        CENTER_Y = y * 2
        center_selected = True
        print(f"원의 중심이 설정되었습니다: X={CENTER_X}, Y={CENTER_Y}")

# 첫 번째 프레임 읽어오기
ret, first_frame = cap.read()
if not ret:
    print("Error: 영상을 읽을 수 없습니다.")
    exit()

preview_window_name = 'Click the Center of the Fringe'
cv2.namedWindow(preview_window_name)
cv2.setMouseCallback(preview_window_name, get_mouse_click)

while not center_selected:
    display_frame = cv2.resize(first_frame.copy(), (width // 2, height // 2))
    
    cv2.putText(display_frame, "Click the exact center of the circles.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow(preview_window_name, display_frame)
    
    # 클릭할 때까지 대기 (1ms마다 키 입력 확인)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("중심 설정이 취소되었습니다.")
        exit()

# 클릭 완료 후 설정 창 닫기
cv2.destroyWindow(preview_window_name)

# 분석할 최대 반경 재설정 (설정된 중심이 한쪽으로 치우쳐도 화면 밖으로 나가지 않도록 최소 거리 계산)
MAX_RADIUS = min(CENTER_X, width - CENTER_X, CENTER_Y, height - CENTER_Y)
print(f"분석을 시작합니다... (분석 반경: {MAX_RADIUS}px)")

# 3. 프레임별 영상 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 

    # 흑백(Grayscale) 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # cv2.warpPolar() 함수를 사용하여 동심원을 직선으로 변환합니다.
    # CENTER_X, CENTER_Y: 원의 중심
    # MAX_RADIUS: 폴라 좌표의 최대 반경
    # flags: 변환 방법 (WARP_POLAR_LINEAR: 선형 폴라 변환)
    gray_polar = cv2.warpPolar(gray, (MAX_RADIUS, height), (CENTER_X, CENTER_Y), MAX_RADIUS, cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS)

    # 변환된 이미지 gray_polar는 Y축이 각도(0-360), X축이 반경(0-MAX_RADIUS)입니다.
    # 각 각도에서의 밝기 값을 평균 내어 '방사형 평균 프로파일'을 구합니다.
    # axis=0 기준으로 평균을 냅니다 (각도 방향 평균)
    radial_profile = np.mean(gray_polar, axis=0) # X축(반경)에 따른 평균 밝기 프로파일
    
    # 데이터를 기록
    current_time = frame_count / fps
    time_axis.append(current_time)
    radial_profiles_history.append(radial_profile) # 각 프레임의 1D 프로파일을 2D 리스트에 저장
    
    # --- 실시간 모니터링 화면 표시 (50% 축소) ---
    preview_frame = frame.copy()
    # 원의 중심과 최대 반경을 녹색 원으로 표시
    cv2.circle(preview_frame, (CENTER_X, CENTER_Y), 5, (0, 255, 0), -1) # 중심
    cv2.circle(preview_frame, (CENTER_X, CENTER_Y), MAX_RADIUS, (0, 255, 0), 4) # 최대 반경
    cv2.putText(preview_frame, f"Time: {current_time:.2f}s", (40, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    # 50% 축소하여 모니터 정중앙에 잘 보이도록 함
    resized_preview = cv2.resize(preview_frame, (width // 2, height // 2))
    cv2.imshow('Michelson-Morley Video Analysis (Radial Averaging, 50% Split)', resized_preview)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# 4. 결과 출력 및 그래프 시각화
# 시공간 이미지(Space-Time Image)로 시각화
# 각 프레임의 1D 프로파일을 세로로 쌓아 2D 이미지를 만듭니다.
# X축: 반경 (Radius), Y축: 시간 (Time)
spacetime_image = np.array(radial_profiles_history)

print("\n[분석 완료] 최종 결과 시공간 그래프를 생성합니다.")

# 시공간 이미지 시각화
plt.figure(figsize=(10, 8))
plt.imshow(spacetime_image, extent=[0, MAX_RADIUS, max(time_axis) if time_axis else 10, 0], aspect='auto', cmap='magma')
plt.title('Michelson-Morley Experiment - Spacetime Analysis (Radial Intensity)', fontsize=14)
plt.xlabel('Radius (pixels)', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.colorbar(label='Grayscale Intensity (0-255)')
plt.tight_layout()
plt.show()
