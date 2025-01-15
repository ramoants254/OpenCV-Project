import cv2 as cv
import numpy as np
import os
import datetime
import csv
import glob
import torch
from torchvision import transforms
from PIL import Image
from google.colab import drive
from google.colab.output import eval_js
from base64 import b64decode
from IPython.display import display, Image, Javascript
import time

# Mount Google Drive (if using Colab)
drive.mount('/content/drive')

## Camera calibration function
def calibrate_camera(chessboard_images_path, chessboard_size=(9, 6), square_size=25):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    obj_points = []
    img_points = []

    images = glob.glob(chessboard_images_path)
    if not images:
        print("Error: No chessboard images found.")
        return None, None

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            obj_points.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_points.append(corners2)
            cv.drawChessboardCorners(img, chessboard_size, corners2, ret)

    if obj_points and img_points:
        ret, cam_mat, dist_coef, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        if ret:
            print("Camera calibrated successfully.")
            return cam_mat, dist_coef
        else:
            print("Camera calibration failed.")
            return None, None
    else:
        print("No corners detected in images. Calibration failed.")
        return None, None

# Function to write CSV headers for marker data
def write_headers(marker_data_file, alert_update_file):
    if not os.path.exists(marker_data_file):
        with open(marker_data_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            headers = ['Timestamp', 'Hour', 'Marker ID', 'X (cm)', 'Y (cm)', 'Z (cm)', 'Distance (cm)']
            csvwriter.writerow(headers)

    if not os.path.exists(alert_update_file):
        with open(alert_update_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            headers = ['Timestamp', 'Hour', 'Notifications', 'Alerts', 'Warnings']
            csvwriter.writerow(headers)

# Record marker data to CSV
def record_data(marker_id, coordinates, distance, marker_data_file):
    timestamp = datetime.datetime.now().isoformat()
    hour = datetime.datetime.now().hour
    x, y, z = [round(coord, 2) for coord in coordinates]
    distance = round(distance, 2)

    with open(marker_data_file, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([timestamp, hour, marker_id, x, y, z, distance])

# Log alerts and notifications
def log_alert_update(event_type, alert_update_file):
    timestamp = datetime.datetime.now().date()
    hour = datetime.datetime.now().hour
    filename = alert_update_file

    # Read existing data
    existing_data = []
    if os.path.exists(filename):
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            existing_data = list(csvreader)

    # Update or create a new entry for the current hour
    updated = False
    for row in existing_data:
        if row[0] == str(timestamp) and row[1] == str(hour):
            row[event_type] = int(row[event_type]) + 1
            updated = True
            break

    if not updated:
        new_row = [str(timestamp), str(hour), 0, 0, 0]
        new_row[event_type] = 1
        existing_data.append(new_row)

    # Write the updated data back
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(existing_data)

# Movement detection function
def check_for_changes(current_coordinates, marker_id, initial_coordinates, threshold=5):
    if marker_id not in initial_coordinates:
        initial_coordinates[marker_id] = current_coordinates
        return True  # Consider the first detection as a change to log it initially

    movement = np.linalg.norm(current_coordinates - initial_coordinates[marker_id])
    return movement > threshold

# Human Pose Detection using Pretrained Model (Mediapipe Pose)
def detect_pose(frame, model):
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    transform = transforms.Compose([transforms.ToTensor()])
    tensor_image = transform(pil_image).unsqueeze(0).to(device)  # Send to GPU if available

    with torch.no_grad():
        output = model(tensor_image)  # Process on GPU
    return output

# Camera Capture
def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

# Main Function
def main():
    chessboard_images_path = '/content/drive/MyDrive/OpenCV/submission/images/*.jpg'
    marker_data_file = 'marker_data.csv'
    alert_update_file = 'alert_update.csv'
    MARKER_SIZE = 10  # centimeters

    cam_mat, dist_coef = calibrate_camera(chessboard_images_path)
    if cam_mat is None or dist_coef is None:
        print("Calibration failed. Exiting.")
        return

    marker_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    param_markers = cv.aruco.DetectorParameters()

    write_headers(marker_data_file, alert_update_file)

    # Load Human Pose Detection Model (Mediapipe or HRNet) here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('facebookresearch/human-pose-estimation.pytorch', 'pose_hrnet_w32', pretrained=True).to(device)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    initial_coordinates = {}
    entry_time = None
    door_ajar_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from camera.")
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, rejected = cv.aruco.detectMarkers(gray_frame, marker_dict, parameters=param_markers)

        if marker_corners and marker_IDs is not None:
            rVec, tVec, _ = cv.aruco.estimatePoseSingleMarkers(marker_corners, MARKER_SIZE, cam_mat, dist_coef)

            for ids, corners, i in zip(marker_IDs, marker_corners, range(len(marker_IDs))):
                corners = corners[0]  # Flatten corners
                cv.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA)

                x, y, z = tVec[i][0]
                distance = np.sqrt(x**2 + y**2 + z**2)

                if check_for_changes(tVec[i][0], ids[0], initial_coordinates):
                    record_data(ids[0], tVec[i][0], distance, marker_data_file)

                    if x > 5 and entry_time is None:
                        entry_time = datetime.datetime.now()
                        print(f"Activity: Someone entered. Marker ID: {ids[0]}")
                        log_alert_update(3, alert_update_file)
                        door_ajar_time = None

                    elif x < 5 and entry_time is not None:
                        print(f"Activity: Person exited. Marker ID: {ids[0]}")
                        log_alert_update(3, alert_update_file)
                        entry_time = None

                    elif distance > 50 and entry_time is None:
                        if door_ajar_time is None:
                            door_ajar_time = datetime.datetime.now()
                        elif (datetime.datetime.now() - door_ajar_time).total_seconds() > 10:
                            print(f"Notification: Door ajar. Marker ID: {ids[0]}")
                            log_alert_update(2, alert_update_file)
                            door_ajar_time = None

                top_right = corners[0].astype(int)
                cv.putText(frame, f"id: {ids[0]} Dist: {round(distance, 2)}", tuple(top_right), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2, cv.LINE_AA)

                if len(corners) > 3:
                    bottom_left = corners[3].astype(int)
                    cv.putText(frame, f"x:{round(x, 1)} y: {round(y, 1)}", tuple(bottom_left), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2, cv.LINE_AA)

                cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)

        # Process human pose detection
        pose_output = detect_pose(frame, model)
        # Process pose_output for fall detection and other analysis here

        cv.imshow("frame", frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
