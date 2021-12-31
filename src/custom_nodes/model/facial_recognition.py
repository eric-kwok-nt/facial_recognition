"""
Node template for creating custom nodes.
"""

import os
from typing import Any, Dict
import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode
from src.modelling.vggface_model import VGGFace_Model
from src.modelling.knn_model import KNN_Classify
from src.datapipeline.images_to_embeddings import Create_Embeddings
import pdb

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
        super().__init__(config, node_path=node_path, **kwargs)
        embedding_path = './data/embedding.pickle' 
        knn_path = './models/knn.pickle'
        self.VGG_M = VGGFace_Model()
        self.VGG_M.download_model()
        self.model_ = self.VGG_M.build_model()
        self.CE = Create_Embeddings()
        self.knn = KNN_Classify(embedding_path, knn_path)
        self.knn.build_model(fit_knn=False)
        self.threshold = 0.36331658291457286
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

        outputs = self.recognise_faces(inputs["img"], inputs["bboxes"])

        return outputs

    def recognise_faces(self, img, bboxes):
        outputs = {
            "bbox_scores": np.array([]),
            "bbox_labels": np.array([]),
            }
        if len(bboxes) > 0:
            height, width, _ = img.shape
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1*width), int(y1*height), int(x2*width), int(y2*height)
                image_cropped = img[y1:y2, x1:x2]
                embedding = self.CE.get_embeddings(image_cropped, self.model_, BGR=False, augment=False)
                y_pred, y_prob = self.knn.predict(embedding,threshold=self.threshold)
                outputs["bbox_labels"] = np.append(outputs["bbox_labels"], y_pred)
                outputs["bbox_scores"] = np.append(outputs["bbox_scores"], y_prob)
        # The following code segment is here for testing, so that the custom
        # node has something to pass to the next node. Please replace with
        # your correct outputs i.e. "bbox_scores", "bbox_labels"
        # outputs = {"img": img, "bboxes": bboxes}

        # returns: Remember to return dict of "bbox_scores" and "bbox_labels"
        # e.g. {
        #          "bbox_scores": ...,
        #          "bbox_labels": ...,
        #      }

        return outputs
