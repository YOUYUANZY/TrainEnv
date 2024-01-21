import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


class ImageSelector:
    def __init__(self, root, model, pType):
        self.model = model
        self.root = root
        self.ptype = pType
        self.root.title("Image Selector")
        # 图片路径变量
        self.image_path_var1 = tk.StringVar()
        self.image_path_var2 = tk.StringVar()
        # 创建选择图片按钮
        self.btn_select_image1 = tk.Button(root, text="选择图片1", command=self.select_image1)
        self.btn_select_image1.pack(pady=10)
        self.btn_select_image2 = tk.Button(root, text="选择图片2", command=self.select_image2)
        self.btn_select_image2.pack(pady=10)
        # 显示选择的图片
        self.image_label1 = tk.Label(root, text="图片1")
        self.image_label1.pack()
        self.image_label2 = tk.Label(root, text="图片2")
        self.image_label2.pack()
        # 创建传递参数按钮
        self.btn_pass_parameters = tk.Button(root, text="人脸预测预测", command=self.pass_parameters)
        self.btn_pass_parameters.pack(pady=10)

    def select_image1(self):
        image_path = filedialog.askopenfilename(title="选择图片1", filetypes=[("Image files", self.ptype)])
        self.image_path_var1.set(image_path)
        self.display_image(image_path, self.image_label1)

    def select_image2(self):
        image_path = filedialog.askopenfilename(title="选择图片2", filetypes=[("Image files", self.ptype)])
        self.image_path_var2.set(image_path)
        self.display_image(image_path, self.image_label2)

    def display_image(self, image_path, label):
        if image_path:
            image = Image.open(image_path)
            image = image.resize((150, 150), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            label.config(image=photo)
            label.image = photo

    def pass_parameters(self):
        image_path1 = self.image_path_var1.get()
        image_path2 = self.image_path_var2.get()
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2)
        self.model.getFeature(image1, image2)
