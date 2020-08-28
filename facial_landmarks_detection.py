'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore

class FacialLandmarksDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extensions = extensions

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Network could not be initialized, Is this the right path?")
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        # Load the model
        self.core = IECore()
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

        # Add extensions
        if cpu_extension and "CPU" in device:
            self.core.add_extension(self.extensions, self.device)

        supported_layers = self.core.query_network(network=self.model, device_name=self.device)
        layers = self.model.layers.keys()
        for l in layers:
            if l not in supported_layers:
                raise ValueError("Unsupported layers, add more extensions")

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_name = self.input_name
        proc_image = self.preprocess_input(image)
        input_dict = {input_name: proc_image}
        infer_request = self.net.start_async(request_id=0, inputs=input_dict)
        infer_status = infer_request.wait()
        if infer_status == 0:
            output = infer_request.outputs[self.output_name]
            return self.draw.outputs(output, image)

    def check_model(self):
        pass

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image_p = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image_p = image_p.transpose((2, 0, 1))
        image_p = image_p.reshape(1, *image_p.shape)
        return  image_p

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        height = image_shape[0]
        width = image_shape[1]

        x_left,y_left = outputs[0][0] * width, outputs[1][0] * height
        x_right,y_right = outputs[2][0] * width, outputs[3][0] * height

        #box for left eye
        x_left_min = int(x_left - 10)
        x_left_max = int(y_left + 10)
        y_left_min = int(x_left - 10)
        y_left_max = int(y_left + 10)
        cv2.rectangle(image, (x_left_min, x_left_max), (y_left_min, y_left_max), (0,255,0), 2)
        left_eye = image[x_left_min:x_left_max, y_left_min:y_left_max]

        #box for right eye
        x_right_min = int(x_right - 10)
        x_right_max = int(x_right + 10)
        y_right_min = int(x_right - 10)
        y_right_max = int(x_right + 10)
        cv2.rectangle(image, (x_right_min, x_right_max), (y_right_min, y_right_max), (0,255,0), 2)
        right_eye = image[x_right_min:x_right_max, y_right_min:y_right_max]

        #Calculate coordinates
        b_box = []
        outputs = outputs[self.output_name][0]
        x_left_eye = int(outputs[0] * width)
        y_left_eye = int(outputs[1] * height)
        x_right_eye = int(outputs[2] * width)
        y_right_eye = int(outputs[3] * height)
        b_box.append(x_left_eye, y_left_eye, x_right_eye, y_right_eye)

        return left_eye, right_eye, b_box

        return left_eye, right_eye
