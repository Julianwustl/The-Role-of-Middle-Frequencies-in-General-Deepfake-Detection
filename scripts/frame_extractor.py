import os
import cv2
import glob
import dlib
def extract_frames(video_path, out_dir, frames_per_second=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_indices = [int(round(i*fps/frames_per_second)) for i in range(int(total_frames/fps*frames_per_second))]

    frame_saved = 0
    for i in range(total_frames):
        ret, frame = cap.read()

        if i in frame_indices:
            frame_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame{i}.png"
            out_path = os.path.join(out_dir, frame_name)
            cv2.imwrite(out_path, frame)
            frame_saved += 1

    cap.release()

    return frame_saved

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mov")):  # Add or modify your video file extensions
                input_file_path = os.path.join(root, file)
                
                # Create the same structure in the output directory
                relative_path = os.path.relpath(root, input_dir)
                output_sub_dir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_sub_dir):
                    os.makedirs(output_sub_dir)

                extract_frames(input_file_path, output_sub_dir)

def process_directory_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".jpg", ".png")):  # Add or modify your video file extensions
                input_file_path = os.path.join(root, file)
                
                # Create the same structure in the output directory
                relative_path = os.path.relpath(root, input_dir)
                output_sub_dir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_sub_dir):
                    os.makedirs(output_sub_dir)

                detect_and_save_faces(input_file_path, output_sub_dir)


def detect_and_save_faces(image_path, output_path):
    # Load the pre-trained model for face detection
    detector = dlib.get_frontal_face_detector()
    

    # Read the image
    img = cv2.imread(image_path)
        
    # Detect faces in the image
    dets = detector(img)
    if len(dets) == 0:
        return
    rect = dets[0]
    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

    # Crop the face
    cropped_face = img[y:y+h, x:x+w]
    if cropped_face.size == 0:
        return
    # Resize the cropped face to 224x224
    #resized_face = cv2.resize(cropped_face, (224, 224))

    # Save the cropped and resized face
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file_path = os.path.join(output_path, f"{base_name}_face.png")
    cv2.imwrite(output_file_path, cropped_face)

if __name__ == "__main__":
    lst =["youtube"]# ["NeuralTextures", "Deepfakes", "Face2Face"] #"FaceShifter", "FaceSwap", 
    for value in lst:
        input_dir = f"/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data/face_forensics/original_sequences/{value}/c23/images"
        
        output_dir = f"/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data/face_forensics/original_sequences/{value}/c23/faces"

        process_directory_images(input_dir, output_dir)