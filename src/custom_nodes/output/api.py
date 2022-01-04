"""
Node template for creating custom nodes.
"""

import os
from typing import Any, Dict
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Special output node used in runner for API calls. Does not generate any outputs.

    Inputs:
        |img|

    Outputs:
        |pipeline_end|

    Configs:
        None.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        # super().__init__(config, node_path=__name__, **kwargs)
        # super().__init__(config, node_path='/Users/ericlee/data/coding/AIAP/team4/src/custom_nodes/configs/model.facial_recognition', **kwargs)
        node_path = os.path.join(
            os.getcwd(), "src/custom_nodes/configs/output.api"
        )
        super().__init__(config, node_path=node_path, **kwargs)
        

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """

        return {"pipeline_end": True}