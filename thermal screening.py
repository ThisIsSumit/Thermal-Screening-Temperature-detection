import os
import numpy as np
import cv2

base_dir = 'Thermal_Screening_Temperature_Detection-main'
threshold = 200
area_of_box = 700       
min_temp = 100           
font_scale_caution = 1   
font_scale_temp = 0.7    


def convert_to_temperature(pixel_avg):
    # Convert Fahrenheit to Celsius
    return (pixel_avg / 2.25 - 32) * 5/9

def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    heatmap_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_HOT)

    _, binary_thresh = cv2.threshold(heatmap_gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    image_erosion = cv2.erode(binary_thresh, kernel, iterations=1)
    image_opening = cv2.dilate(image_erosion, kernel, iterations=1)

    contours, _ = cv2.findContours(image_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_with_rectangles = np.copy(heatmap)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w * h < area_of_box:
            continue 

        mask = np.zeros_like(heatmap_gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        mean = convert_to_temperature(cv2.mean(heatmap_gray, mask=mask)[0])

        temperature = round(mean, 2)
        color = (0, 255, 0) if temperature < min_temp else (255, 255, 127)

        if temperature >= min_temp:
            cv2.putText(image_with_rectangles, "High temperature detected !!!", (35, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_caution, color, 2, cv2.LINE_AA)

        image_with_rectangles = cv2.rectangle(
            image_with_rectangles, (x, y), (x+w, y+h), color, 2)

        cv2.putText(image_with_rectangles, "{} C".format(temperature), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_temp, color, 2, cv2.LINE_AA)

    return image_with_rectangles

def main():
    
    video = cv2.VideoCapture(0)
    video_frames = []

    while True:
        ret, frame = video.read()

        if not ret:
            print("Error reading frame from the camera.")
            break

        frame = process_frame(frame)
        video_frames.append(frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    if video_frames:
       
        out_path = 'full/path/to/output.avi' 
        size = (video_frames[0].shape[1], video_frames[0].shape[0])

        try:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(out_path, fourcc, 30, size)

            for i in range(len(video_frames)):
                out.write(video_frames[i])

            out.release()

            print(f"Video saved successfully to {out_path}")

        except Exception as e:
            print(f"Error saving video: {e}")

    else:
        print("No frames were processed. Check if the camera is connected and working properly.")

if __name__ == "__main__":
    main()
