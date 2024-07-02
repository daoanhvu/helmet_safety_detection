from ultralytics import YOLO


MODEL_PATH = 'yolov10n.pt'
model = YOLO(MODEL_PATH)


def do_detect_and_show(img_path):
    results = model(source=img_path)
    # results[0].show()
    # for index, row, in detections.iterrows:
    #     print(row)


if __name__ == '__main__':
    # IMAGE_PATH = 'D:\\projects\\helmet_safety_detection\\dataset\\test\\test3.jpg'
    # do_detect_and_show(IMAGE_PATH)

    VIDEO_PATH = 'https://youtu.be/wqPSsu7XQ74'
    do_detect_and_show(VIDEO_PATH)
