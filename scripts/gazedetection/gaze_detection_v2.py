#
# Jetson Nano - Real-time Gaze Tracker (Based on Enhanced iTracker[1])
#
# [1] Gaze-tracking as Accessibility Technology: A Deep Learning Approach (arXiv:2010.05123v1)
#
# This is a *functionally complete* script, including a full
# implementation of the 'preprocess' function based on the paper.
#

import cv2
import dlib
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  
import time
import math


# --- 1. Constants (USER MUST CONFIGURE) ---
ENGINE_PATH = "enhanced_itracker_fp16.plan"
LANDMARK_MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
INPUT_SHAPE_HW = (224, 224) # H, W for the model inputs (e.g., 224x224)
GRID_SIZE = 25 # W & H for the face grid

# Model input names (must match the names used during ONNX export)
INPUT_NAME_EYE_L = "eye_L"
INPUT_NAME_EYE_R = "eye_R"
INPUT_NAME_FACE = "face"
INPUT_NAME_GRID = "grid"

# Model output name (must match)
OUTPUT_NAME_GAZE = "gaze_xy"

# Landmark indices for Dlib's 68-point model
LEFT_EYE_INDICES = list(range(36, 42))
RIGHT_EYE_INDICES = list(range(42, 48))
NOSE_INDICES = list(range(27, 36))
FACE_OUTLINE_INDICES = list(range(0, 17))



# --- 2. TensorRT Helper Class ---
# (This class handles the complex boilerplate of PyCUDA and TensorRT)

class TrtModel:
    """Simplified TensorRT Inference wrapper."""
    
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Allocate host (CPU) and device (GPU) buffers
        self.h_inputs = {}
        self.d_inputs = {}
        self.h_outputs = {}
        self.d_outputs = {}
        self.bindings = []

        for binding in self.engine:
            name = self.engine.get_binding_name(binding)
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host (CPU) memory
            h_mem = np.empty(shape, dtype=dtype)
            # Allocate device (GPU) memory
            d_mem = cuda.mem_alloc(h_mem.nbytes)
            
            self.bindings.append(int(d_mem))
            
            if self.engine.binding_is_input(binding):
                self.h_inputs[name] = h_mem
                self.d_inputs[name] = d_mem
            else:
                self.h_outputs[name] = h_mem
                self.d_outputs[name] = d_mem
                
        print(f"TensorRT Engine loaded. Inputs: {self.h_inputs.keys()}, Outputs: {self.h_outputs.keys()}")

    def predict(self, inputs_dict):
        """
        Run inference.
        'inputs_dict' is a dictionary like:
        {'eye_L': np_array, 'eye_R': np_array, ...}
        """
        
        # 1. Copy inputs from Host (CPU) to Device (GPU)
        for name, data in inputs_dict.items():
            if name not in self.h_inputs:
                print(f"Warning: Input '{name}' not found in model.")
                continue
            
            # Ensure data matches the host buffer
            np.copyto(self.h_inputs[name], data.ravel())
            cuda.memcpy_htod_async(self.d_inputs[name], self.h_inputs[name], self.stream)

        # 2. Run Inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 3. Copy outputs from Device (GPU) to Host (CPU)
        for name in self.h_outputs.keys():
            cuda.memcpy_dtoh_async(self.h_outputs[name], self.d_outputs[name], self.stream)

        # 4. Synchronize stream
        self.stream.synchronize()

        # 5. Return outputs
        return self.h_outputs


# --- 3. Preprocessing Functions (IMPLEMENTED) ---

def get_landmark_points(landmarks):
    """Convert dlib landmarks object to a numpy array."""
    return np.array([(p.x, p.y) for p in landmarks.parts()], dtype=np.float32)

def get_rotation_matrix(landmark_pts):
    """
    Calculate rotation matrix to correct head rotation (Paper Sec 4.6).
    We use the eyes and nose bridge to find the rotation.
    """
    # Use points from eyes and nose bridge as stable points
    left_eye_center = landmark_pts[LEFT_EYE_INDICES].mean(axis=0)
    right_eye_center = landmark_pts[RIGHT_EYE_INDICES].mean(axis=0)
    
    # Calculate angle
    eye_delta_y = right_eye_center[1] - left_eye_center[1]
    eye_delta_x = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(eye_delta_y, eye_delta_x))
    
    # Get center of rotation (center of the face landmarks)
    rotation_center = landmark_pts.mean(axis=0)
    
    # Create rotation matrix
    M = cv2.getRotationMatrix2D((rotation_center[0], rotation_center[1]), angle, 1.0)
    return M, rotation_center

def transform_points(points, M):
    """Apply an affine transformation (rotation) to landmark points."""
    # Add a column of ones for matrix multiplication
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    # Apply transformation
    transformed_points = (M @ points_homogeneous.T).T
    return transformed_points.astype(np.float32)

def crop_roi(frame, points, scale=1.0):
    """
    Crop a Region of Interest (ROI) from the frame.
    Finds the center of the points, calculates a bounding box based on the
    max distance from the center, scales it, and crops a square region.
    """
    center = points.mean(axis=0)
    # Find max distance from center to any point
    max_dist = np.max(np.linalg.norm(points - center, axis=1))
    # Scale and make it square
    size = int(max_dist * 2 * scale)
    
    half_size = size // 2
    x_min = int(center[0] - half_size)
    y_min = int(center[1] - half_size)
    x_max = x_min + size
    y_max = y_min + size
    
    # Ensure coordinates are within frame bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame.shape[1], x_max)
    y_max = min(frame.shape[0], y_max)
    
    # Crop and handle edge cases
    crop = frame[y_min:y_max, x_min:x_max]
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        return None
        
    return crop

def create_face_grid(frame_shape, face_rect, grid_size=GRID_SIZE):
    """
    Create the 25x25 face grid input (from iTracker baseline).
    This grid indicates the position of the face in the frame.
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    # Get face bounding box
    x_min = face_rect.left()
    y_min = face_rect.top()
    x_max = face_rect.right()
    y_max = face_rect.bottom()
    
    # Normalize coordinates to [0, 1]
    norm_x_min = x_min / frame_shape[1]
    norm_y_min = y_min / frame_shape[0]
    norm_x_max = x_max / frame_shape[1]
    norm_y_max = y_max / frame_shape[0]
    
    # Scale coordinates to grid size
    grid_x_min = int(norm_x_min * grid_size)
    grid_y_min = int(norm_y_min * grid_size)
    grid_x_max = int(math.ceil(norm_x_max * grid_size))
    grid_y_max = int(math.ceil(norm_y_max * grid_size))
    
    # Fill the grid (1s where the face is)
    grid[grid_y_min:grid_y_max, grid_x_min:grid_x_max] = 1.0
    
    # Reshape for the model: (1, 1, H, W)
    grid = np.expand_dims(grid, axis=0) # (1, H, W)
    grid = np.expand_dims(grid, axis=0) # (1, 1, H, W)
    return grid

def normalize_and_format(image, target_size_hw):
    """
    Final processing for an image patch:
    1. Convert to YCrCb (Paper Sec 4.5)
    2. Resize to target (e.g., 224x224)
    3. Normalize to [0, 1]
    4. Transpose from (H, W, C) to (C, H, W)
    5. Add batch dimension (1, C, H, W)
    """
    # 1. Color Transformation (BGR to YCrCb)
    try:
        img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    except cv2.error:
        # Handle cases where crop might be empty
        return None
        
    # 2. Resize
    img_resized = cv2.resize(img_ycrcb, (target_size_hw[1], target_size_hw[0]))
    
    # 3. Normalize
    # *Note: This assumes normalization to [0, 1]. If your model was
    # trained with ImageNet stats, you must use that normalization instead.*
    img_norm = (img_resized / 255.0).astype(np.float32)
    
    # 4. Transpose (H, W, C) -> (C, H, W)
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    
    # 5. Add batch dimension (1, C, H, W)
    img_final = np.expand_dims(img_transposed, axis=0)
    
    return img_final


def preprocess(frame, landmarks_dlib, face_rect):
    """
    Full preprocessing pipeline.
    
    Returns:
        A dictionary of preprocessed numpy arrays for the model, or None.
    """
    
    # 1. Get 68 landmark points as numpy array
    landmark_pts = get_landmark_points(landmarks_dlib)

    # 2. Perform Rotation Correction (Paper Sec 4.6)
    M, rot_center = get_rotation_matrix(landmark_pts)
    frame_h, frame_w = frame.shape[:2]
    rotated_frame = cv2.warpAffine(frame, M, (frame_w, frame_h))
    
    # 3. Get new landmark coordinates on the rotated frame
    rotated_landmark_pts = transform_points(landmark_pts, M)

    # 4. Crop Eye/Face Regions (Paper Sec 4.6)
    # We use a scaling factor to get a wider, more informative crop
    # (These scales are typical; you may need to tune them)
    eye_scale = 1.5
    face_scale = 1.2
    
    eye_L_crop = crop_roi(rotated_frame, rotated_landmark_pts[LEFT_EYE_INDICES], scale=eye_scale)
    eye_R_crop = crop_roi(rotated_frame, rotated_landmark_pts[RIGHT_EYE_INDICES], scale=eye_scale)
    face_crop = crop_roi(rotated_frame, rotated_landmark_pts, scale=face_scale) # Crop based on all landmarks

    if eye_L_crop is None or eye_R_crop is None or face_crop is None:
        print("Warning: Crop failed (ROI out of bounds).")
        return None

    # 5. Resize, Normalize, and Format
    eye_L_final = normalize_and_format(eye_L_crop, INPUT_SHAPE_HW)
    eye_R_final = normalize_and_format(eye_R_crop, INPUT_SHAPE_HW)
    face_final = normalize_and_format(face_crop, INPUT_SHAPE_HW)
    
    if eye_L_final is None or eye_R_final is None or face_final is None:
        print("Warning: Normalization failed (empty crop).")
        return None

    # 6. Create Face Grid (from iTracker baseline)
    # Note: The grid is based on the *original* (un-rotated) face rectangle
    grid_final = create_face_grid(frame.shape, face_rect, grid_size=GRID_SIZE)

    # 7. Return the dictionary
    return {
        INPUT_NAME_EYE_L: eye_L_final,
        INPUT_NAME_EYE_R: eye_R_final,
        INPUT_NAME_FACE: face_final,
        INPUT_NAME_GRID: grid_final
    }


# --- 4. GStreamer Camera Pipeline ---
# (Omitted for brevity, same as previous version)
def get_jetson_gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
    """Returns a GStreamer pipeline string for Jetson Nano."""
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# --- 5. Main Execution ---

def main():
    print("Loading Dlib models...")
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(LANDMARK_MODEL_PATH)
    except Exception as e:
        print(f"Error loading Dlib models: {e}")
        print(f"Make sure '{LANDMARK_MODEL_PATH}' exists.")
        return

    print("Loading TensorRT engine...")
    try:
        gaze_model = TrtModel(ENGINE_PATH)
    except Exception as e:
        print(f"Error loading TensorRT engine: {e}")
        print(f"Make sure '{ENGINE_PATH}' exists and was built for this Jetson.")
        return

    # --- [REMOVED] Network Socket Setup ---
    # ... (networking code removed) ...
    # ----------------------------------

    print("Opening camera...")
    # Use 0 for a standard USB webcam
    # Or use get_jetson_gstreamer_pipeline() for CSI camera
    cap = cv2.VideoCapture(0) 
    # cap = cv2.VideoCapture(get_jetson_gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting main loop...")
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # For performance, we can shrink the frame for detection
        # small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        # gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Detect Face
        faces = detector(gray)
        
        if len(faces) > 0:
            # Use only the first detected face
            face_rect = faces[0]
            
            # (If using a small_frame, scale face_rect back to original)
            # face_rect = dlib.rectangle(face.left()*2, face.top()*2, face.right()*2, face.bottom()*2)
            
            # 2. Detect Landmarks
            landmarks = predictor(gray, face_rect)
            
            # 3. Preprocess Frame (THE IMPLEMENTED FUNCTION)
            inputs_dict = preprocess(frame, landmarks, face_rect)
            
            if inputs_dict is not None:
                # 4. Run Inference
                outputs = gaze_model.predict(inputs_dict)
                
                # 5. Postprocess (Get gaze vector)
                gaze_xy = outputs[OUTPUT_NAME_GAZE] # This is a 1D array, e.g., [x, y]
                x_cm, y_cm = gaze_xy[0], gaze_xy[1]
                
                # --- Visualization (Example) ---
                print(f"Gaze (cm): x={x_cm:.2f}, y={y_cm:.2f}")

                # --- [MODIFIED] 6. Calculate Quadrant ---
                # A. Calculate Quadrant (Interpreting "q1 q2 q3 q4")
                #    This assumes (0,0) is the screen center.
                #    (Note: The paper's (x,y) might need offset/scaling)
                quadrant = "center"
                if x_cm > 0.5 and y_cm > 0.5:
                    quadrant = "q1" # Top-Right (example)
                elif x_cm < -0.5 and y_cm > 0.5:
                    quadrant = "q2" # Top-Left (example)
                elif x_cm < -0.5 and y_cm < -0.5:
                    quadrant = "q3" # Bottom-Left (example)
                elif x_cm > 0.5 and y_cm < -0.5:
                    quadrant = "q4" # Bottom-Right (example)

                # B. Print the result (instead of sending)
                print(f"Quadrant: {quadrant}")
                
                # (You can now call other functions with this 'quadrant' variable)
                # my_other_function(quadrant, x_cm, y_cm)
                
                # ----------------------------------------------------
                
                # Draw landmarks on the frame
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Calculate FPS
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow("Jetson Nano Gaze Tracker", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Releasing resources...")

    cap.release()
    cv2.destroyAllWindows()