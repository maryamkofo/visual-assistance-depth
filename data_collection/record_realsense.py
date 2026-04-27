import pyrealsense2 as rs
import numpy as np
import cv2
import os


os.makedirs("dataset/rgb", exist_ok=True)
os.makedirs("dataset/depth", exist_ok=True)

# Set up RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)


align = rs.align(rs.stream.color)

frame_idx = 0
print("Recording... Press 'q' to stop.")

try:
    while True:
       
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

     
        if not color_frame or not depth_frame:
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())  # in millimeters

        # Save RGB image and depth map
        cv2.imwrite(f"dataset/rgb/{frame_idx:05d}.png", color_image)
        np.save(f"dataset/depth/{frame_idx:05d}.npy", depth_image)

        frame_idx += 1

       
        cv2.imshow("RGB", color_image)

        # Visualize depth as a colormap for preview
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        cv2.imshow("Depth", depth_colormap)

        # Press q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Stopped. Saved {frame_idx} frames.")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
