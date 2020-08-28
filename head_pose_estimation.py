'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
import cv2
import logging as log
import os
from openvino.inference_engine import IENetwork, IECore

class HeadPoseEstimation:
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
        return image_p

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        output = []
        output.append(outputs['angle_y_fc'][0][0])
        output.append(outputs['angle_p_fc'][0][0])
        output.append(outputs['angle_r_fc'][0][0])
        return output


