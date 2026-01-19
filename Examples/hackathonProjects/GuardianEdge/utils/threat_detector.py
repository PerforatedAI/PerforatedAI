"""
Threat Detection Logic
"""

import torch
import numpy as np


class ThreatDetector:
    """
    Manages threat detection and alerting logic
    """
    
    def __init__(self, threat_classes, conf_threshold=0.5):
        """
        Initialize threat detector
        
        Args:
            threat_classes: List of class IDs considered threats
            conf_threshold: Confidence threshold for detections
        """
        self.threat_classes = set(threat_classes)
        self.conf_threshold = conf_threshold
        self.threat_history = []
        
    def check_threats(self, detections):
        """
        Check if any threats are present in detections
        
        Args:
            detections: YOLO detection results (boxes)
        
        Returns:
            List of threat detections
        """
        threats = []
        
        if detections is None or len(detections) == 0:
            return threats
        
        # Extract class IDs and confidences
        if hasattr(detections, 'cls') and hasattr(detections, 'conf'):
            classes = detections.cls.cpu().numpy()
            confidences = detections.conf.cpu().numpy()
            boxes = detections.xyxy.cpu().numpy()
            
            for i, (cls, conf, box) in enumerate(zip(classes, confidences, boxes)):
                if int(cls) in self.threat_classes and conf >= self.conf_threshold:
                    threat = {
                        'class_id': int(cls),
                        'confidence': float(conf),
                        'bbox': box.tolist(),
                        'timestamp': None  # Can add timestamp if needed
                    }
                    threats.append(threat)
        
        # Update threat history
        if threats:
            self.threat_history.extend(threats)
            # Keep only recent threats (last 100)
            self.threat_history = self.threat_history[-100:]
        
        return threats
    
    def get_threat_summary(self):
        """
        Get summary of recent threats
        
        Returns:
            Dictionary with threat statistics
        """
        if not self.threat_history:
            return {
                'total_threats': 0,
                'threats_by_class': {}
            }
        
        threats_by_class = {}
        for threat in self.threat_history:
            cls = threat['class_id']
            if cls not in threats_by_class:
                threats_by_class[cls] = 0
            threats_by_class[cls] += 1
        
        return {
            'total_threats': len(self.threat_history),
            'threats_by_class': threats_by_class
        }
    
    def clear_history(self):
        """Clear threat history"""
        self.threat_history = []
