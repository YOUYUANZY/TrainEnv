# import cv2
import tkinter as tk
from FaceNet import FaceNet
from imageSelectGUI import ImageSelector


def predict(config):
    if config.type == 'facenet':
        FR = FaceNet(config=config)
    else:
        raise ValueError('unsupported predict type')
    root = tk.Tk()
    app = ImageSelector(root, FR, config.photoType)
    root.geometry("400x600")
    root.mainloop()
