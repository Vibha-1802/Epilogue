import math

class SensorFusionNode:
    def __init__(self):
        pass
        
    def process_boxes(self, boxes, frame_width):
        """Converts bounding boxes into mock cartesian space objects"""
        fused_objects = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            
            w = x2 - x1
            mock_dist = max(3.0, 1500.0 / (w + 1.0)) 
            img_center_x = frame_width / 2
            obj_center_x = (x1 + x2) / 2
            mock_azimuth = (obj_center_x - img_center_x) / img_center_x * 30.0 
            mock_rate = -5.0 
            
            azimuth_rad = math.radians(mock_azimuth)
            x_current = mock_dist * math.cos(azimuth_rad)
            y_current = mock_dist * math.sin(azimuth_rad)
            vx = mock_rate * math.cos(azimuth_rad)
            vy = mock_rate * math.sin(azimuth_rad)
            
            fused_objects.append({
                'obj_id': i,
                'class_id': cls_id,
                'x': x_current,
                'y': y_current,
                'vx': vx,
                'vy': vy
            })
        return fused_objects
