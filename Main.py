import cv2
import os
import numpy as np
import logging as log
import sys
import math
import time
from argparse import ArgumentParser
from face_detection import FaceDetection
from  facial_landmarks_detection import FacialLandmarksDetection
from gaze_estimation import GazeEstimation
from head_pose_estimation import HeadPoseEstimation
from mouse_controller import MouseController
from input_feeder import InputFeeder

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--face_detection_model", required=True, type=str,
                        help="Path to a face detection model xml file")
    parser.add_argument("-fl", "--facial_landmarks_model", required=True, type=str,
                        help="Path to a facial landmarks detection model xml file")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str,
                        help="Path to a head pose estimation model xml file")
    parser.add_argument("-ge", "--gaze_estimation_model", required=True, type=str,
                        help="Path to a gaze estimation model xml file")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or CAM")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    parser.add_argument("-fp", "--frame_out", required=False, nargs='+',
                        default=[],
                        help="Example: --fp fd fl hp ge (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame,"
                             "fd for Face Detection, fl for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation.")
    return parser

#def check_input(stream):




def infer_on_stream(args):

    logger = log.getLogger()
    input_feed = args.input
    input_feeder = None

    if input_feed == 'CAM':
        input_feeder = InputFeeder('cam')

    else:
        if not os.path.isfile(input_feed):
            logger.error("Unable to find video file")
            exit(1)
        else:
          input_feeder = InputFeeder('video', input_feed)

    mouse_controller = MouseController('medium', 'fast')
    input_feeder.load_data()


    #Initialize models
    face_detection = FaceDetection(args.face_detection_model, args.device, args.prob_threshold, args.cpu_extension)
    facial_landmarks_detection = FacialLandmarksDetection(args.facial_landmarks_model, args.device, args.cpu_extension)
    head_pose_estimation = HeadPoseEstimation(args.head_pose_model, args.device, args.cpu_extension)
    gaze_estimation = GazeEstimation(args.gaze_estimation_model, args.device, args.cpu_extension)

    start_time = time.time()

    #Loading Models
    face_detection.load_model()
    facial_landmarks_detection.load_model()
    head_pose_estimation.load_model()
    gaze_estimation.load_model()

    end_time = time.time() - start_time

    #mouse_controller = MouseController('medium', 'fast')

    #Handle input stream
    #input_feed = check_input(args.input)
    #feeder = InputFeeder(input_feed=input_feed, input_file = args.input)
    #feeder.load_data()

    frame_count = 0
    f_detection = 0
    l_detection = 0
    hp_detection = 0
    ge_detection = 0

    frame_out = args.frame_out


    for ret, frame in input_feeder.next_batch():
        if not ret:
            break
        key_pressed = cv2.waitKey(60)

        frame_count += 1
        start_time = time.time()
        face_coords,face = face_detection.predict(frame.copy())
        f_detection += time.time() - start_time

        if len(face_coords) == 0:
            log.error("Face could not be detected.")
            if key == 27:
                break
            continue

        start_time = time.time()
        left_eye_image, right_eye_image, eye_coord = facial_landmarks_detection.predict(face)
        l_detection += time.time() - start_time

        start_time = time.time()
        headpose_output = head_pose_estimation.predict(face)
        hp_detection += time.time() - start_time

        start_time = time.time()
        mouse_coord, gaze_vector = gaze_estimation.predict(left_eye_image, right_eye_image, headpose_output)
        ge_detection += time.time() - start_time

        if frame_out:
            f_preview = frame.copy()
            if 'fd' in frame_out:
                f_preview = face.copy()
                cv2.rectangle(f_preview, (face_coord[0], face_coord[1]),(face_coord[2], face_coord[3]), (0, 255, 0), 2)
            
            if 'fl' in frame_out:
                f_preview = face.copy()
                cv2.rectangle(f_preview, (eye_coord[0][0], eye_coord[0][1]), (eye_coord[0][2], eye_coord[0][3]), (180, 0, 180))
                cv2.rectangle(f_preview, (eye_coord[1][0], eye_coord[1][1]), (eye_coord[1][2], eye_coord[1][3]), (180, 0, 180))

            if 'hp' in frame_out:
                cv2.putText(f_preview,"Pose Angle: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(headpose_out[0],headpose_out[1],headpose_out[2]),(0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

            if 'ge' in frame_out:
                arrow_x = gaze_vector[0] * 12
                arrow_y = gaze_vector[1] * 12

                cv2.arrowedLine(f_preview, (eye_coord[0][0], eye_coord[0][1]), int(eye_coord[0][2] + arrow_x), int(eye_coord[0][3] + arrow_y), (0, 255, 0), 2)
                cv2.arrowedLine(f_preview, (eye_coord[0][0], eye_coord[0][1]), int(eye_coord[0][2] + arrow_x), int(eye_coord[0][3] + arrow_y), (0, 255, 0), 2)

            if len(f_preview) != 0:
                img_cap = np.hstack((cv2.resize(frame, (700, 700)), cv2.resize(f_preview, (700, 700))))
            else:
                img_cap = cv2.resize(frame, (700, 700))

            cv2.imshow('Screenview', img_cap)
            mouse_controller.move(mouse_coords[0], mouse_coords[1])

            if key_pressed == 27:
                break

    if frame_count > 0:
        logging.info("Face Detection:{:.1f}ms".format(1000 * f_detection / frame_count))
        logging.info("Facial Landmarks Detection:{:.1f}ms".format(1000 * l_detection / frame_count))
        logging.info("Headpose Estimation:{:.1f}ms".format(1000 * hp_detection / frame_count))
        logging.info("Gaze Estimation:{:.1f}ms".format(1000 * ge_detection / frame_count))

    logger.error("Stream ended...")
    cv2.destroyAllWindows()
    input_feeder.close()

def main():

    args = build_argparser().parse_args()
    infer_on_stream(args)

if __name__ == '__main__':
    main()