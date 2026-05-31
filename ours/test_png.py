import cv2, numpy as np
img = cv2.imread(r"C:\anomaly_detection\src\results_private\can\test_private_heatmaps\000_regular_heatmap.png")  # BGR, 3 kênh
b, g, r = cv2.split(img)
print("Có phải grayscale?", np.array_equal(b, g) and np.array_equal(g, r))