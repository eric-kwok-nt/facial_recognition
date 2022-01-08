import os
from peekingduck.runner import Runner
from peekingduck.pipeline.nodes.input import live, recorded
from peekingduck.pipeline.nodes.model import mtcnn
from .custom_nodes.model import facial_recognition
from .custom_nodes.output import api, custom_output
from peekingduck.pipeline.nodes.dabble import fps
from peekingduck.pipeline.nodes.draw import bbox, legend
from peekingduck.pipeline.nodes.output import screen
import argparse


def runner(
    type="live_video", 
    input_filepath="./data/raw/videos/test", 
    input_source=0,
    save_video_path='./data/raw/videos/saved/saved.avi',
    fps_=10
    ) -> None:
    """Runs the Peeking Duck pipeline.

    Args:
        type (str, optional): Whether to use live webcam video or from a recorded video, or from a live webcam video and saving the recorded frames as a video file. Defaults to "live_video".
        input_filepath (str, optional): Path to the input video file. Applicable for type='recorded_video'. Defaults to "./data/raw/videos/test".
        input_source (int, optional): Input source integer value. Refer to cv2 VideoCapture class. Applicable for type=['live_video' | 'live_video_and_save']. Defaults to 0.
        save_video_path (str, optional): Path for video to be saved. Applicable for type='live_video_and_save'. Defaults to './data/raw/videos/saved'.
        fps_ (int, optional): Frames per second for video to be saved. Applicable for type='live_video_and_save'. Defaults to 20.

    Returns:
        None
    """
    # Initialise the nodes
    if type in ["live_video", 'live_video_and_save']:
        input_node = live.Node(input_source=input_source)  # get images from webcam
    elif (type == "recorded_video") or (type == "api"):
        input_node = recorded.Node(
            input_dir=os.path.join(os.getcwd(), input_filepath),
            threading=True,
            buffer_frames=True,
        )  # get images from local file

    mtcnn_node = mtcnn.Node()  # face detection
    model_node = facial_recognition.Node()  # facial recognition
    if type != "api":
        dabble_node = fps.Node()  # frames per second
    draw_bbox_node = bbox.Node(show_labels=True)  # draw bounding boxes
    draw_legend_node = legend.Node()  # display fps in legend box
    if (type == "live_video") or (type == "recorded_video"):
        output_node = screen.Node()  # display output to screen
    elif type == "api":
        output_node = api.Node()  # no outputs except return values to API
    elif type == 'live_video_and_save':
        output_node = custom_output.Node(video_path=save_video_path, fps=fps_)

    # Don't have Dabble:FPS node if it's for an API call
    if type == "api":
        runner_node = Runner(
            nodes=[
                input_node,
                mtcnn_node,
                model_node,
                draw_bbox_node,
                draw_legend_node,
                output_node,
            ]
        )
    else:
        runner_node = Runner(
            nodes=[
                input_node,
                mtcnn_node,
                model_node,
                dabble_node,
                draw_bbox_node,
                draw_legend_node,
                output_node,
            ]
        )
    runner_node.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial Recoginition algorithm")
    parser.add_argument(
        '--type', type=str, default='live_video',
        choices=['live_video', 'recorded_video', 'live_video_and_save'],
        help="Whether to use live webcam video or from a recorded video, or from a live webcam video and saving the recorded frames as a video file.")
    parser.add_argument(
        '--input_filepath', type=str, default="./data/raw/videos/test",
        help="The path to your video files if --type is 'recorded_video'")
    parser.add_argument(
        '--input_source', type=int, default=0,
        help="Input source integer value. Refer to cv2 VideoCapture class. Applicable for --type ['live_video' | 'live_video_and_save']")
    parser.add_argument(
        '--save_video_path', type=str, default="./data/raw/videos/saved/saved.avi",
        help="Path for video to be saved. Applicable for --type 'live_video_and_save'")
    parser.add_argument(
        '--fps', type=int, default=10,
        help="Frames per second for video to be saved. Applicable for --type 'live_video_and_save'")
    parsed = parser.parse_args()
    runner(
        type=parsed.type, 
        input_filepath=parsed.input_filepath, 
        input_source=parsed.input_source,
        save_video_path=parsed.save_video_path,
        fps_=parsed.fps
        )
