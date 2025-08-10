from ultralytics import YOLO
import cv2
import numpy as np
import functions as func
import os

# CUSTOM FUNCTIONS

os.makedirs('models', exist_ok=True)
model=YOLO('models/yolo11l-pose.pt')
image_name='image_3.jpg'
predict=model(source=f'images/{image_name}')

annotated_frame=predict[0].plot()



# For each detected person:

if predict[0].keypoints is not None:
    for kpts in predict[0].keypoints.data:
        # Extract points (in format x,y,score)
        kpts = kpts.cpu().numpy()

        LEFT_SHOULDER = kpts[5][:2]
        LEFT_ELBOW = kpts[7][:2]
        LEFT_WRIST = kpts[9][:2]

        if np.all(LEFT_SHOULDER) and np.all(LEFT_ELBOW) and np.all(LEFT_WRIST):
            angle = func.calculate_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
            rula_score = func.get_rula_score(angle)
            risk = func.get_risk_label(rula_score)

            # Show on frame
            cv2.putText(annotated_frame, f"Angle: {int(angle)}°", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"RULA: {rula_score}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(annotated_frame, f"Risk: {risk}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


os.makedirs('outputs', exist_ok=True)
cv2.imshow("Ergonomic Posture Detection (YOLO-Pose)", annotated_frame)
cv2.imwrite(f"outputs/{image_name}", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()