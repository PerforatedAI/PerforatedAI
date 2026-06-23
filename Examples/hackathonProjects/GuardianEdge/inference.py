"""
GuardianEdge Inference Script
Real-time threat detection using PAI-optimized YOLO model
"""

import argparse
import yaml
import cv2
import torch
import time
from pathlib import Path
from ultralytics import YOLO

# PerforatedAI imports
from perforatedai import utils_perforatedai as UPA

from utils.threat_detector import ThreatDetector


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_inference(config, args):
    """
    Run real-time inference with threat detection
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    print("="*60)
    print("GuardianEdge - Real-Time Threat Detection")
    print("="*60)
    
    # Load model
    model_path = args.model if args.model else "models/best_model_pai.pt"
    print(f"\n[1/4] Loading model: {model_path}")
    
    if args.compare:
        print("   Loading baseline YOLO for comparison...")
        baseline_model = YOLO(args.baseline if args.baseline else 'yolov8n.pt')
    
    # For PAI models, load using UPA if it's a pure PAI save
    # Otherwise use YOLO's loader
    try:
        model = YOLO(model_path)
        print("   Model loaded successfully!")
    except Exception as e:
        print(f"   Error loading model: {e}")
        print("   Attempting to load as PAI model...")
        # Load as PAI model (requires base model structure)
        # This would need proper implementation based on saved format
        raise
    
    # Initialize threat detector
    print(f"[2/4] Initializing threat detector...")
    detector = ThreatDetector(
        threat_classes=config['detection']['threat_classes'],
        conf_threshold=config['detection']['confidence_threshold']
    )
    print(f"   Monitoring classes: {config['detection']['threat_classes']}")
    print(f"   Confidence threshold: {config['detection']['confidence_threshold']}")
    
    # Setup video source
    source = args.source
    if source == '0':
        source = 0  # Webcam
    print(f"\n[3/4] Opening video source: {source}")
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Resolution: {width}x{height} @ {fps} FPS")
    
    # Setup output
    if config['inference']['save_output']:
        output_dir = Path(config['inference']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_dir / 'output.mp4'),
            fourcc, fps, (width, height)
        )
    
    print(f"\n[4/4] Starting detection loop...")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'p' - Pause/Resume")
    print()
    
    frame_count = 0
    total_time = 0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or read error")
                    break
                
                # Run inference
                start_time = time.time()
                results = model(frame, verbose=False)
                inference_time = time.time() - start_time
                
                # Process results
                for result in results:
                    # Draw bounding boxes
                    annotated_frame = result.plot(
                        line_width=config['inference']['line_thickness'],
                        labels=config['inference']['show_labels'],
                        conf=config['inference']['show_conf']
                    )
                    
                    # Check for threats
                    detections = result.boxes
                    if detections is not None and len(detections) > 0:
                        threats = detector.check_threats(detections)
                        if threats:
                            # Draw threat alert on frame
                            cv2.rectangle(annotated_frame, (10, 10), (width-10, 80),
                                        (0, 0, 255), -1)
                            cv2.putText(annotated_frame, "THREAT DETECTED!",
                                      (30, 50), cv2.FONT_HERSHEY_BOLD,
                                      1.5, (255, 255, 255), 3)
                
                # Calculate FPS
                frame_count += 1
                total_time += inference_time
                avg_fps = frame_count / total_time if total_time > 0 else 0
                
                # Display metrics
                metrics_text = f"FPS: {avg_fps:.1f} | Inference: {inference_time*1000:.1f}ms"
                cv2.putText(annotated_frame, metrics_text, (10, height-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('GuardianEdge - Threat Detection', annotated_frame)
                
                # Save if enabled
                if config['inference']['save_output']:
                    out.write(annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                screenshot_path = f"screenshot_{int(time.time())}.jpg"
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
        if config['inference']['save_output']:
            out.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("Inference Complete")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average inference time: {(total_time/frame_count)*1000:.2f}ms" if frame_count > 0 else "N/A")
        print()


def main():
    parser = argparse.ArgumentParser(description='Run GuardianEdge inference')
    
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (default: models/best_model_pai.pt)')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (0 for webcam, path to video file, or image dir)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with baseline YOLO model')
    parser.add_argument('--baseline', type=str, default='yolov8n.pt',
                        help='Baseline model for comparison')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run inference
    run_inference(config, args)


if __name__ == '__main__':
    main()
