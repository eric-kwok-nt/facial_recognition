import os
from peekingduck.runner import Runner
from peekingduck.pipeline.nodes.input import live, recorded
from peekingduck.pipeline.nodes.model import mtcnn
from .custom_nodes.model import facial_recognition
from peekingduck.pipeline.nodes.dabble import fps
from peekingduck.pipeline.nodes.draw import bbox, legend
from peekingduck.pipeline.nodes.output import screen

recorded_video_filepath = "./data/raw/videos/test"

def runner(live_video=True):
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
    if live_video:
        input_node = live.Node(input_source=1)  # get images from webcam
    else:
        input_node = recorded.Node(
            input_dir=os.path.join(os.getcwd(), recorded_video_filepath),
            threading=True,
            buffer_frames=True,
        )  # get images from local file

    mtcnn_node = mtcnn.Node()  # face detection
    model_node = facial_recognition.Node()  # facial recognition
    dabble_node = fps.Node()  # frames per second
    draw_bbox_node = bbox.Node(show_labels=True)  # draw bounding boxes
    draw_legend_node = legend.Node()  # display fps in legend box
    output_node = screen.Node()  # display output to screen

    # Run it in the runner
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

    # Inspect the data
    # print(f"type(runner.pipeline.data): {type(runner.pipeline.data)}")
    # print(f"runner.pipeline.data: {runner.pipeline.data}")

    return (
        runner.pipeline.data["img"],
        runner.pipeline.data["bboxes"],
        runner.pipeline.data["bbox_labels"],
    )


if __name__ == "__main__":
    runner(live_video=True)
