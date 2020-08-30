'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import numpy as np
import cv2
import logging as log
import time
import sys
from openvino.inference_engine import IENetwork, IECore

class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold=0.60, extensions=None):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.extensions = extensions
        self.threshold = threshold
        self.device = device
        self.core = None
        self.net = None

        try:
            self.core = IECore
            self.model = self.core.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Network could not be initialized, Is this the right path?")
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        #Load the model
        self.core = IECore()
        self.model = self.core.read_network(self.model_structure, self.model_weights)

        #Add extensions
        if self.extensions and "CPU" in device:
            self.core.add_extension(self.extensions, self.device)

        supported_layers= self.core.query_network(network=self.model, device_name=self.device)
        layers = self.model.layers.keys()
        for l in layers:
            if l not in supported_layers:
                raise ValueError("Unsupported layers, add more extensions")
                
        self.net = self.core.read_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, image):
        input_name = self.input_name
        proc_image = self.preprocess_input(image)
        input_dict = {input_name:proc_image}
        infer_request = self.net.start_async(request_id=0, inputs=input_dict)
        infer_status = infer_request.wait()
        if infer_status == 0:
            output = infer_request.outputs[self.output_name]
            return self.draw.outputs(output, image)

    def check_model(self):
        pass

    def preprocess_input(self, image):
        image_p = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image_p = image_p.transpose((2, 0, 1))
        image_p = image_p.reshape(1, *image_p.shape)
        return image_p


    def preprocess_output(self, outputs, image):
        width = image.shape[1]
        height = image.shape[0]
        face_box =[]
        for box in outputs[0][0]:
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                face_box.append((xmin, ymin, xmax, ymax))
        return face_box, image


