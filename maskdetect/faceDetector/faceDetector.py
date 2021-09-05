from . import openpifpaf

class Detector:
    """
    Detector class is a high level class for detecting object using x86 devices.
    When an instance of the Detector is created you can call inference method and feed your
    input image in order to get the detection results.
    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """
    def __init__(self, gpu_num):
        self.net = openpifpaf.OpenPPWrapper(gpu_num)
        self.width = self.net.w
        self.height = self.net.h 
        
    def inference(self, resized_rgb_image):
        # self.fps = self.net.fps
        output = self.net.inference(resized_rgb_image)
        return output
