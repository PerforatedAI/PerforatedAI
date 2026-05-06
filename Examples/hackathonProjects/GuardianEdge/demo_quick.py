"""
GuardianEdge Quick Demo (Baseline YOLO - No Training Required)
This demo uses a pretrained YOLO model to show threat detection capabilities
"""

import argparse
import cv2
import time
from ultralytics import YOLO

# Define threat classes (COCO dataset)
THREAT_CLASSES = {
    0: "person",
    39: "bottle",
    43: "knife",
    44: "fork",
    46: "banana"  # Using as test object
}


def run_demo(source='0', model_name='yolov8n.pt'):
    """
    Run GuardianEdge demo with baseline YOLO
    
    Args:
        source: Video source (0 for webcam, path to file, or image directory)
        model_name: YOLO model to use
    """
    print("="*60)
    print("GuardianEdge - Quick Demo (Baseline YOLO)")
    print("="*60)
    print(f"\nLoading model: {model_name}")
    
    # Load pretrained YOLO model
    model = YOLO(model_name)
    print("Model loaded successfully!")
    
    # Setup video source
    if source == '0':
        source = 0  # Webcam
    
    print(f"\nOpening video source: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'")
        print("\nFor webcam (source=0), make sure:")
        print("  - Your webcam is connected")
        print("  - No other app is using it")
        print("\nFor video files:")
        print("  - Provide full path to video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    
    print("\nStarting detection...")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'p' - Pause/Resume")
    print()
    
    frame_count = 0
    total_time = 0
    paused = False
    threat_count = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or read error")
                    break
                
                # Run inference
                start_time = time.time()
                results = model(frame, verbose=False, conf=0.5)
                inference_time = time.time() - start_time
                
                # Process results
                for result in results:
                    # Draw bounding boxes
                    annotated_frame = result.plot(line_width=2)
                    
                    # Check for threats
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        detected_threats = []
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            if cls_id in THREAT_CLASSES:
                                detected_threats.append(f"{THREAT_CLASSES[cls_id]} ({conf:.2f})")
                                threat_count += 1
                        
                        if detected_threats:
                            # Draw threat alert
                            alert_height = 80
                            cv2.rectangle(annotated_frame, (10, 10), (width-10, alert_height),
                                        (0, 0, 255), -1)
                            cv2.putText(annotated_frame, "⚠ THREAT DETECTED ⚠",
                                      (30, 50), cv2.FONT_HERSHEY_BOLD,
                                      1.2, (255, 255, 255), 3)
                            
                            # Show detected threats
                            y_pos = alert_height + 30
                            for threat in detected_threats[:3]:  # Show max 3
                                cv2.putText(annotated_frame, threat,
                                          (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                          0.6, (0, 0, 255), 2)
                                y_pos += 25
                
                # Calculate FPS
                frame_count += 1
                total_time += inference_time
                avg_fps = frame_count / total_time if total_time > 0 else 0
                
                # Display metrics
                metrics_text = f"FPS: {avg_fps:.1f} | Inference: {inference_time*1000:.1f}ms | Threats: {threat_count}"
                cv2.putText(annotated_frame, metrics_text, (10, height-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add "DEMO MODE" watermark
                cv2.putText(annotated_frame, "DEMO MODE (Baseline YOLO)", (10, height-50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Show frame
                cv2.imshow('GuardianEdge Demo - Threat Detection', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                screenshot_path = f"demo_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"Screenshot saved: {screenshot_path}")
            elif key == ord('p'):
                paused = not paused
                status = "Paused" if paused else "Resumed"
                print(f"{status}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("Demo Complete")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average inference time: {(total_time/frame_count)*1000:.2f}ms" if frame_count > 0 else "N/A")
        print(f"Total threats detected: {threat_count}")
        print()


def main():
    parser = argparse.ArgumentParser(description='GuardianEdge Quick Demo')
    
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (0 for webcam, path to video file)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO model to use (default: yolov8n.pt)')
    
    args = parser.parse_args()
    
    run_demo(source=args.source, model_name=args.model)


if __name__ == '__main__':
    main()
