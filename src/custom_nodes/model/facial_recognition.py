"""
Node template for creating custom nodes.
"""

import os
from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        # super().__init__(config, node_path=__name__, **kwargs)
        # super().__init__(config, node_path='/Users/ericlee/data/coding/AIAP/team4/src/custom_nodes/configs/model.facial_recognition', **kwargs)
        node_path = os.path.join(
            os.getcwd(), "src/custom_nodes/configs/model.facial_recognition"
        )
        print(f"node_name: {self.node_name}")
        print(f"config: {self.config}")
        super().__init__(config, node_path=node_path, **kwargs)

        # initialize/load any configs and models here
        # configs can be called by self.<config_name> e.g. self.filepath
        # self.logger.info(f"model loaded with configs: config")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """

        # testing: see if we are getting the bounding boxes from the inputs
        #print(inputs["bboxes"])

        self.recognise_faces(inputs["img"], inputs["bboxes"])

        outputs = {
            "bbox_scores": ...,
            "bbox_labels": ...,
        }

        return outputs

    def recognise_faces():
        # put your facial recognition function here
        pass

        # returns: "bbox_scores" and "bbox_labels"

        