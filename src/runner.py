from peekingduck.runner import Runner
from peekingduck.pipeline.nodes.input import live
from peekingduck.pipeline.nodes.model import mtcnn
from src.custom_nodes.model import facial_recognition
from peekingduck.pipeline.nodes.draw import bbox
from peekingduck.pipeline.nodes.output import screen

# Initialise the nodes
input_node = live.Node()
mtcnn_node = mtcnn.Node()
model_node = facial_recognition.Node()
draw_node = bbox.Node()
output_node = screen.Node()

# Run it in the runner
runner = Runner(nodes=[input_node, mtcnn_node, model_node, draw_node, output_node])
runner.run()

#Inspect the data
# runner.pipeline.data

printed = False
if not printed:
    print(f"type(runner.pipeline.data): {type(runner.pipeline.data)}")
    print(f"runner.pipeline.data: {runner.pipeline.data}")
    printed = True
