from peekingduck.runner import Runner
from peekingduck.pipeline.nodes.input import live
from peekingduck.pipeline.nodes.model import mtcnn
from .custom_nodes.model import facial_recognition
from peekingduck.pipeline.nodes.draw import bbox
from peekingduck.pipeline.nodes.output import screen

def runner():
    # Initialise the nodes
    input_node = live.Node()                # get images from webcam
    mtcnn_node = mtcnn.Node()               # face detection
    model_node = facial_recognition.Node()  # facial recognition
    draw_node = bbox.Node()                 # draw bounding boxes
    output_node = screen.Node()             # display output to screen

    # Run it in the runner
    runner = Runner(nodes=[input_node, mtcnn_node, model_node, draw_node, output_node])
    runner.run()

    # Inspect the data
    print(f"type(runner.pipeline.data): {type(runner.pipeline.data)}")
    print(f"runner.pipeline.data: {runner.pipeline.data}")

    return runner.pipeline.data['img'], runner.pipeline.data['bboxes'], runner.pipeline.data['bbox_labels']

if __name__ == '__main__':
    runner()