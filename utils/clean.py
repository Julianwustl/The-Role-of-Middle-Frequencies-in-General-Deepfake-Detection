import av
import cv2
from pytorchvideo.data.encoded_video import EncodedVideo

def clean_video(input_path, output_path):
    try:
        input_container = av.open(input_path)
        output_container = av.open(output_path, "w")
        in_stream = input_container.streams.video[0]
        out_stream = output_container.add_stream(template=in_stream)

        for packet in input_container.demux(in_stream):
            print(packet)

            # We need to skip the "flushing" packets that `demux` generates.
            if packet.dts is None:
                continue

            # We need to assign the packet to the new stream.
            packet.stream = out_stream

            output_container.mux(packet)

        input_container.close()
        output_container.close()
        print(f"Cleaned video saved at {output_path}")
    except av.AVError as e:
        print(f"Failed to clean the video at {input_path}: {e}")

def repair_video(input_file, output_file):
    # Open the corrupted video file
    cap = cv2.VideoCapture(input_file)
    
    # Get video details
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        # Read next frame
        ret, frame = cap.read()
        if ret:
            # If the frame is successfully read, write it to the new file
            out.write(frame)
        else:
            # If the frame is not successfully read (i.e., it's corrupted),
            # just skip it and move on to the next frame
            continue

    # Release everything after the job is finished
    cap.release()
    out.release()
    print(f"Repaired video saved at {output_file}")

def repair_video_av(input_file, output_file):
    input_container = av.open(input_file)
    output_container = av.open(output_file, 'w')

    input_stream = input_container.streams.video[0]
    output_stream = output_container.add_stream(template=input_stream)
    
    for packet in input_container.demux(input_stream):
        try:
            if packet.dts is None:
                continue
            for frame in packet.decode():
                packet.stream = output_stream.encode(frame)
                if packet:
                    output_container.mux(packet)
        except av.AVError:
            continue  # Skip corrupted frame

    # Flush stream
    packet = output_stream.encode(None)
    if packet:
        output_container.mux(packet)

    output_container.close()


if __name__ == "__main__":
    import glob
    import os

    input_path = "/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data/face_forensics/original_sequences/**/*.mp4"
    output = "/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data/face_forensics/clean/"
    for path in glob.glob(input_path, recursive=True)[:10]:
        video_pyav = EncodedVideo.from_path(path,decode_audio=False, decoder='pyav')
        print(video_pyav.duration)
        print(video_pyav.get_clip(0, 1))
        #output_path = path.split("/")[-1].replace(".mp4", "_cleaned.mp4")
        #if os.path.exists(output_path):
        #    continue
        #repair_video(path, output + output_path)
    # for video in glob.glob("/home/wustl/VideoClassifier/data/Emotion/Train/*/*"):
    #     output_path = video.replace(".mp4", "_cleaned.mp4")
    #     if os.path.exists(output_path):
    #         continue
    #     clean_video(video, output_path)
