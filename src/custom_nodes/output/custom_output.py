"""
Shows the outputs on your display.
"""

from typing import Any, Dict

import cv2
import os
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Streams the output on your display.

    Inputs:
        |img|

    Outputs:
        |pipeline_end|

    Configs:
        None.
    """

    def __init__(
        self, config: Dict[str, Any] = None, 
        video_path='./data/raw/videos/saved/saved.avi', 
        fps=10, 
        **kwargs: Any
        ) -> None:
        
        node_path = os.path.join(
            os.getcwd(), "src/custom_nodes/configs/output.custom_output"
        )
        super().__init__(config, node_path=node_path, **kwargs)
        self.frames = []
        self.video_path = video_path
        self.fps = fps

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Show the outputs on your display, press 'q' to stop and save video"""
        cv2.imshow("PeekingDuck", inputs["img"])
        self.frames.append(inputs['img'])

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.save_to_video()
            return {"pipeline_end": True}

        return {"pipeline_end": False}
    
    def save_to_video(self):
        assert self.video_path is not None, "Please initialize custom_output node with video_path!"
        folder, _ = os.path.split(self.video_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        h, w, _ = self.frames[0].shape
        out = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, (w,h))
        for i in range(len(self.frames)):
            out.write(self.frames[i])
        out.release()