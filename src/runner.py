import os
from peekingduck.runner import Runner
from peekingduck.pipeline.nodes.input import live, recorded
from peekingduck.pipeline.nodes.model import mtcnn
from .custom_nodes.model import facial_recognition
from .custom_nodes.output import api
from peekingduck.pipeline.nodes.dabble import fps
from peekingduck.pipeline.nodes.draw import bbox, legend
from peekingduck.pipeline.nodes.output import screen
import argparse


def runner(type="live_video", input_filepath="./data/raw/videos/test", input_source=0):
    """Runs the Peeking Duck pipeline.

    Args:
        live_video (bool, optional): Set this to True if we are using live
        webcam video. If this is False, the input will be a recorded video
        in the 'recorded_video_filepath' directory. Defaults to True.

    Returns:
        runner.pipeline.data["img"],
        runner.pipeline.data["bboxes"],
        runner.pipeline.data["bbox_labels"]: The Peeking Duck data pool dict items.
    """
    # Initialise the nodes
    if type == "live_video":
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

    # Don't have Dabble:FPS node if it's for an API call
    if type == "api":
        runner = Runner(
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
        runner = Runner(
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
    runner.run()

    return (
        runner.pipeline.data["img"],
        runner.pipeline.data["bboxes"],
        runner.pipeline.data["bbox_labels"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial Recoginition algorithm")
    parser.add_argument(
        '--type', type=str, default='live_video',
        choices=['live_video', 'recorded_video'])
    parser.add_argument(
        '--input_filepath', type=str, default="./data/raw/videos/test",
        help="The path to your video files if --type is 'recorded_video'")
    parser.add_argument(
        '--input_source', type=int, default=0,
        help='Input source integer value. Refer to cv2 VideoCapture class')
    parsed = parser.parse_args()
    runner(
        type=parsed.type, 
        input_filepath=parsed.input_filepath, 
        input_source=parsed.input_source)
