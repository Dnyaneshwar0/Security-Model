import os
import sys
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from modules.unauthorized_access.inference import UnauthorizedAccessModule

def run_with_webcam(module):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = module.run(frame)
        print(f"[{result['status'].upper()}] - {result['details']}")
        annotated = result.get("annotated_frame", frame)
        cv2.imshow("Unauthorized Access Monitoring", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_with_videos(module):
    input_dir = "videodata"
    output_dir = "outputdata"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith((".mp4", ".avi", ".mov")):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, f"output_{file}")
            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                print(f"❌ Could not open {file}")
                continue

            width = int(cap.get(3))
            height = int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            print(f"📽️ Processing {file}...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = module.run(frame)
                annotated = result.get("annotated_frame", frame)
                out.write(annotated)

            cap.release()
            out.release()
            print(f"✅ Output saved to {output_path}")

if __name__ == "__main__":
    module = UnauthorizedAccessModule(config={"window_area": (100, 500, 50, 400)})
    print("1. Use Webcam")
    print("2. Process videos in 'videodata/' folder")
    choice = input("Enter your choice: ")

    if choice == '2':
        run_with_videos(module)
    else:
        run_with_webcam(module)
