import torch
import torchvision
import os

# --- 1. 필수 파일 경로 ---
# 이 스크립트는 'L2CSNet_gaze360.pkl' 파일이
# 이 스크립트와 "같은 폴더"에 있다고 가정합니다.
PKL_FILENAME = 'L2CSNet_gaze360.pkl'
ONNX_FILENAME = 'l2cs_gaze360.onnx'

# --- 2. 모델 아키텍처 불러오기 ---
# (같은 폴더에 있는 'model.py' 파일을 찾습니다)
try:
    from model import L2CS
    print("Found 'model.py' (AI architecture)...")
except ModuleNotFoundError:
    print("Error: 'model.py' not found.")
    print("Please make sure this script ('onnx_export.py') is inside the 'l2cs' folder,")
    print("right next to 'model.py'.")
    exit()

# --- 3. .pkl 모델 파일 확인 ---
if not os.path.exists(PKL_FILENAME):
    print(f"Error: Model file '{PKL_FILENAME}' not found in this folder.")
    print("Please move the 'L2CSNet_gaze360.pkl' file into this folder")
    print("(inside the 'l2cs' folder, right next to this script).")
    exit()
else:
    print(f"Found model file: '{PKL_FILENAME}'")

# --- 4. 모델 구조 정의 (ResNet50 백본) ---
print("Initializing model architecture (ResNet50, 28 bins)...")

# ResNet50 모델 로드 (최신 torchvision 경고 수정)
backbone = torchvision.models.resnet50(weights=None)

# [FIX 1] AttributeError: 'expansion' 수정
# (model.py가 이 속성을 필요로 하므로 수동으로 추가)
backbone.expansion = 4 
print("Applied 'backbone.expansion' fix.")

# [FIX 2] TypeError: 'num_bins' 수정
# (model.py에 3개의 인자 전달: backbone, arch_name, num_bins)
model = L2CS(backbone, 'ResNet50', 28)
print("Model architecture created.")

# --- 5. 훈련된 가중치(.pkl) 로드 ---
print(f"Loading pre-trained weights from '{PKL_FILENAME}'...")
loaded_data = torch.load(PKL_FILENAME)

# .pkl 파일이 딕셔너리('model_state_dict') 안에 저장된 경우
if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
    model.load_state_dict(loaded_data['model_state_dict'])
    print("Loaded model state_dict from .pkl dictionary.")
# .pkl 파일이 가중치(state_dict)를 직접 저장한 경우
else:
    model.load_state_dict(loaded_data)
    print("Loaded model state_dict directly.")

model.eval() # 추론(evaluation) 모드로 설정

# --- 6. ONNX 변환을 위한 더미 입력 생성 ---
dummy_input = torch.randn(1, 3, 224, 224)
print("Dummy input tensor created.")

# --- 7. ONNX로 변환 ---
print(f"Exporting model to '{ONNX_FILENAME}'...")
torch.onnx.export(model,
                  dummy_input,
                  ONNX_FILENAME,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['pitchyaw'])

print("="*50)
print(f"Success! Model has been converted to {ONNX_FILENAME}")
print("You can now copy this .onnx file to your Jetson Nano and convert it using trtexec.")
print("="*50)