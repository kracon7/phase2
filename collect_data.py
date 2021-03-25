import os
import time
import argparse
import pickle
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys

plt.ion()

def main(args):

    # if not os.path.isdir(args.output_dir):
    #     os.makedirs(args.output_dir)

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    time_start = time.time()

    color_data_dir = os.path.join(args.output_dir, 'color')
    os.system('mkdir -p ' + color_data_dir)
    depth_data_dir = os.path.join(args.output_dir, 'depth')
    os.system('mkdir -p ' + depth_data_dir)

    fig, ax = plt.subplots(2,1)

    index = 0

    # Streaming loop
    try:
        while time.time() - time_start < args.timeout:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            index += 1

            if index > 1:
                print("saving image {}".format(index))

                color_filename = os.path.join(color_data_dir, '{}.png'.format(index))
                plt.imsave(color_filename, color_image, vmin=0, vmax=255)

                depth_filename = os.path.join(depth_data_dir, '{}.mat'.format(index))
                np.save(depth_filename, depth_image)

                if args.visualize:
                    ax[0].imshow(color_image)
                    ax[1].imshow(depth_image)
                    plt.pause(0.01)

            # time.sleep(1)
    finally:
        pipeline.stop()
                
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--timeout', default=20, type=float, help='total time of image streaming')
    parser.add_argument('--output_dir', default='data', help='directory to store images')
    parser.add_argument('--visualize', type=int, default='0', help='whether visualize the collected data')
    args = parser.parse_args()
    
    main(args)