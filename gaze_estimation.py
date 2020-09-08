'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
import cv2
import logging as log
import math
from openvino.inference_engine import IENetwork, IECore


class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extensions = extensions
        #self.core = None
        #self.model = None
        #self.network = None
        #self.net = None

        try:
            3self.core = IECore()
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
        self.network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

        # Add extensions
        if self.extensions and "CPU" in device:
            self.core.add_extension(self.extensions, self.device)

        supported_layers = self.core.query_network(network=self.model, device_name=self.device)
        layers = self.model.layers.keys()
        for l in layers:
            if l not in supported_layers:
                raise ValueError("Unsupported layers, add more extensions")

    def predict(self, left_eye, right_eye, head_position):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        proc_left_eye = self.preprocess_input(left_eye, self.input_shape)
        proc_right_eye = self.preprocess_input(right_eye, self.input_shape)
        input_dict = {'image_left':proc_left_eye, 'image_right':proc_right_eye, 'head_pose_angle':head_position}
        infer_request = self.net.start_async(request_id=0, inputs=input_dict )
        infer_status = infer_request.wait()
        if infer_status == 0:
            output = infer_request.outputs[self.output_name]
            mouse_coord, gaze_vector = self.preprocess_output(output, head_position)

            return mouse_coord, gaze_vector


    def check_model(self):
        pass

    def preprocess_input(self, left_eye, right_eye):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        left_eye_p = cv2.resize(left_eye, (self.input_shape[3], self.input_shape[2]))
        left_eye_p = left_eye_p.transpose((2,0,1))
        left_eye_p = left_eye_p.reshape(1, *left_eye_p.shape)

        right_eye_p = cv2.resize(right_eye, (self.input_shape[3], self.input_shape[2]))
        right_eye_p = right_eye_p.transpose((2,0,1))
        right_eye_p = right_eye_p.reshape(1, *right_eye_p.shape)

        return  left_eye_p, right_eye_p



    def preprocess_output(self, outputs, head_pose_output):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        gaze_vector = outputs[self.output_name][0]
        roll_value = head_pose_output[2]
        cos_phi = math.cos(roll_value * math.pi / 180)
        sin_phi = math.sin(roll_value * math.pi / 180)

        value_x = gaze_vector[0] * cos_phi + gaze_vector[1] * sin_phi
        value_y = gaze_vector[1] * cos_phi + gaze_vector[0] * sin_phi

        return  (value_x, value_y), gaze_vector
