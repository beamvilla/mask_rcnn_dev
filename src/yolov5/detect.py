from typing import List, Tuple
import torch
import sys
sys.path.append("./")

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import (
    Profile, 
    check_img_size, 
    non_max_suppression, 
    scale_boxes, 
    xyxy2xywh,
    scale_segments
)
from yolov5.utils.segment.general import masks2segments, process_mask
from yolov5.utils.dataloaders import LoadImages


class YOLODetection:
    def __init__(
        self, 
        weight_path: str,
        dnn: bool = True,
        fp16: bool = False,
        mask: bool = False
    ) -> None:
        self.BATCH_SIZE = 1

        self.weight_path = weight_path
        self.dnn = dnn
        self.fp16 = fp16
        self.mask = mask
        self.device = select_device("")
        self.load_model()

    def load_model(self) -> None:
        self.model = DetectMultiBackend(
                        weights=self.weight_path, 
                        device=self.device, 
                        dnn=self.dnn, 
                        data=None, 
                        fp16=self.fp16
                    )
        
    def detect(self,
        image_path: str,
        image_size: List[int] = [640, 640],
        conf_thres: float = 0.25,
        iou_thres: float = 0.5,
        max_det: int = 20
    ) -> Tuple[List[float], List[float], List[int]]:
        stride, pt = self.model.stride, self.model.pt
        image_size = check_img_size(image_size, s=stride)  # check image size
        dataset = LoadImages(
                    image_path, 
                    img_size=image_size, 
                    stride=stride, 
                    auto=pt, 
                    vid_stride=1
                )
        # Run inference
        dt = (Profile(), Profile(), Profile())
        for _, im, im0s, _, _ in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                if self.mask:
                    pred, proto = self.model(im, augment=False, visualize=False)[:2]
                else:
                    pred = self.model(im, augment=False, visualize=False)

            # NMS
            with dt[2]:
                if not self.mask:
                    pred = non_max_suppression(
                                prediction=pred, 
                                conf_thres=conf_thres, 
                                iou_thres=iou_thres, 
                                max_det=max_det
                            )
                else:
                    pred = non_max_suppression(
                                prediction=pred, 
                                conf_thres=conf_thres, 
                                iou_thres=iou_thres, 
                                max_det=max_det,
                                nm=32
                            )
        
        bbox = []
        conf_scores = []
        classes = []
        masks = []
        segments = []

        for i, det in enumerate(pred):  # per image
            im0 = im0s.copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                if self.mask:
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    segments = [
                        scale_segments(im.shape[2:], x, im0.shape, normalize=True)
                        for x in reversed(masks2segments(masks))]
                _det = det
                if self.mask:
                    _det = det[:, :6]
                for *xyxy, conf, cls in reversed(_det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    bbox.append(xywh)
                    conf_scores.append(conf.item())
                    classes.append(int(cls.item()))
        if self.mask:
            return bbox, segments, conf_scores, classes
        return bbox, conf_scores, classes