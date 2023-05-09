import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage
import numpy
from math import log10, sqrt
from statistics import mean

file_path = ""
#wyliczenie współczynnika PSNR
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
#wyliczenie VIF
def vifp_mscale(ref, dist):
    sigma_nsq = 2
    eps = 1e-10

    num = 0.0
    den = 0.0
    for scale in range(1, 5):

        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += numpy.sum(numpy.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += numpy.sum(numpy.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den

    if numpy.isnan(vifp):
        return 1.0
    else:
        return vifp

#detekcja twarzy
def detect_face(frame):
   net = cv2.dnn.readNetFromCaffe("weights-prototxt.txt", "res_ssd_300Dim.caffeModel")

   (height, width) = frame.shape[:2]
   blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                               (300, 300), (104.0, 177.0, 123.0))

   net.setInput(blob)
   detections = net.forward()

   faces = []
   for i in range(0, detections.shape[2]):
       confidence = detections[0, 0, i, 2]

       if confidence < 0.5:
           continue

       box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
       (x1, y1, x2, y2) = box.astype("int")

       faces.append((x1, y1, x2, y2))
   return faces
# Odczywywanie pliku wideo
video = cv2.VideoCapture(file_path)
width= int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))



# sprawdzanie czy wideo się otworzyło
if video.isOpened():
    print("Video opened successfully")
else:
    print("Error opening video")
    exit()

# nastawienie detektora
detector = cv2.xfeatures2d.SIFT_create()

# ustawienie algorytmu do detekcji cech
matcher = cv2.BFMatcher(cv2.NORM_L2)

#zapis pliku
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width,height))


PSNR_values = []
MAE_values = []
VIF_val= []
smoothness_values = []
prev_points = None

while True:
    ret, frame = video.read()
    if not ret:
        break

    # konwersja do skali szarosci
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detekcja cech
    kp, desc = detector.detectAndCompute(gray, None)
    points = np.array([p.pt for p in kp], dtype=np.float32)

    if prev_points is not None:
        matches = matcher.match(desc, prev_desc)
        src_pts = np.array([points[m.queryIdx] for m in matches], dtype=np.float32)
        dst_pts = np.array([prev_points[m.trainIdx] for m in matches], dtype=np.float32)
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        compensated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),flags=cv2.INTER_CUBIC)
        difference_frame = cv2.absdiff(frame, compensated_frame)

        difference_frame = cv2.cvtColor(difference_frame, cv2.COLOR_BGR2GRAY)

        smoothness = np.mean(difference_frame)

        smoothness_values.append(smoothness)
    else:
        # jeśli pierwsza klatka nie wykrywaj cech
        compensated_frame = frame

    # twarz w skompensowanym obrazie
    faces = detect_face(compensated_frame)
    #centrowanie twarzy
    if len(faces) > 0:
        x1, y1, x2, y2 = faces[0]
        x, y, w, h = x1, y1, x2 - x1, y2 - y1

        face_center = (x + w // 2, y + h // 2)

        M = np.float32([[1, 0, frame.shape[1] // 2 - face_center[0]],
                        [0, 1, frame.shape[0] // 2 - face_center[1]]])
        compensated_frame = cv2.warpAffine(compensated_frame, M, (frame.shape[1], frame.shape[0]))

    # wyliczanie miar oceny jakości
    #PSNR = cv2.PSNR(frame, compensated_frame)
    #VIF = vifp_mscale(frame, compensated_frame)
    #MAE = np.mean(np.abs(compensated_frame-frame))

    # Append the values to the lists
    #PSNR_values.append(PSNR)
    ##VIF_val.append(VIF)
    #MAE_values.append(MAE)
    #frame_concat = np.concatenate((frame, compensated_frame), axis=1)
    frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cframe_resized = cv2.resize(compensated_frame, (0, 0), fx=0.5, fy=0.5)

    # Show the original and compensated frames
    cv2.imshow("Original", frame_resized)
    cv2.imshow("Compensated", cframe_resized)
    out.write(compensated_frame)

    # Update the previous frame and points
    prev_gray = gray
    prev_points = points
    prev_desc = desc

    # Wait for user input and update the display
    key = cv2.waitKey(10)
    if key == 27:
        break

#print("PSNR ",mean(PSNR_values))
#print("MAE ",mean(MAE_values))
#print("VIF ",mean(VIF_val))
#plt.figure()

#Plot the values over time
#plt.plot(PSNR_values, label="VIF")
#plt.plot(MAE_values, label="MAE")
#plt.legend()

#plt.show()
# Add a legend and show the plot
#plt.figure()
#plt.legend()
#plt.plot(MAE_values, label="MAE")
#plt.plot(smoothness_values)
#plt.legend()

#plt.show()

# Release the video capture object and destroy the windows
out.release()
video.release()
cv2.destroyAllWindows()



