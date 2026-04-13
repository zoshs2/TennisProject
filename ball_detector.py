from tracknet import BallTrackerNet
import torch
import cv2
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

class BallDetector:
    def __init__(
        self,
        path_model=None,
        device='cuda',
        max_dist=80,
        heatmap_thresh=127,
        min_radius=2,
        max_radius=7,
    ):
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.device = device
        self.max_dist = max_dist
        self.heatmap_thresh = heatmap_thresh
        self.min_radius = min_radius
        self.max_radius = max_radius
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
        self.width = 640
        self.height = 360

    def infer_model(self, frames):
        """ Run pretrained model on a consecutive list of frames
        :params
            frames: list of consecutive video frames
        :return
            ball_track: list of detected ball points
        """
        ball_track = [(None, None)]*2
        prev_pred = [None, None]
        with torch.no_grad():
            for num in tqdm(range(2, len(frames))):
                img = cv2.resize(frames[num], (self.width, self.height))
                img_prev = cv2.resize(frames[num-1], (self.width, self.height))
                img_preprev = cv2.resize(frames[num-2], (self.width, self.height))
                imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
                imgs = imgs.astype(np.float32)/255.0
                imgs = np.rollaxis(imgs, 2, 0)
                inp = np.expand_dims(imgs, axis=0)

                out = self.model(torch.from_numpy(inp).float().to(self.device))
                output = out.argmax(dim=1).detach().cpu().numpy()
                x_pred, y_pred = self.postprocess(output, prev_pred)
                prev_pred = [x_pred, y_pred]
                ball_track.append((x_pred, y_pred))
        return ball_track

    def postprocess(self, feature_map, prev_pred, scale=2):
        """
        :params
            feature_map: feature map with shape (1,360,640)
            prev_pred: [x,y] coordinates of ball prediction from previous frame
            scale: scale for conversion to original shape (720,1280)
            max_dist: maximum distance from previous ball detection to remove outliers
        :return
            x,y ball coordinates
        """
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, self.heatmap_thresh, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=self.min_radius,
                                   maxRadius=self.max_radius)
        x, y = None, None
        if circles is not None:
            if prev_pred[0] is not None and prev_pred[1] is not None:
                best_point = None
                best_dist = np.inf
                for i in range(len(circles[0])):
                    x_temp = circles[0][i][0]*scale
                    y_temp = circles[0][i][1]*scale
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < best_dist:
                        best_dist = dist
                        best_point = (x_temp, y_temp)
                if best_point is not None and best_dist < max_dist:
                    x, y = best_point
            else:
                x = circles[0][0][0]*scale
                y = circles[0][0][1]*scale
        return x, y
