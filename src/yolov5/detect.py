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
    xyxy2xywh
)
from yolov5.utils.dataloaders import LoadImages


class YOLODetection:
    def __init__(
        self, 
        image_path: str,
        weight_path: str,
        image_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.5,
        max_det: int = 20,
        dnn: bool = True,
        fp16: bool = False
    ) -> None:
        self.BATCH_SIZE = 1

        self.image_path = image_path
        self.weight_path = weight_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.dnn = dnn
        self.fp16 = fp16
        self.device = select_device("")
        self.image_size = image_size
        self.load_model()

    def load_model(self) -> None:
        self.model = DetectMultiBackend(
                        weights=self.weight_path, 
                        device=self.device, 
                        dnn=self.dnn, 
                        data=None, 
                        fp16=self.fp16
                    )
        
    def detect(self):
        stride, pt = self.model.stride, self.model.pt
        self.image_size = check_img_size(self.image_size, s=stride)  # check image size
        dataset = LoadImages(
                    self.image_path, 
                    img_size=self.image_size, 
                    stride=stride, 
                    auto=pt, 
                    vid_stride=1
                )

        # Run inference
        #model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
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
                pred = self.model(im, augment=False, visualize=False)

            # NMS
            with dt[2]:
                pred = non_max_suppression(
                            prediction=pred, 
                            conf_thres=self.conf_thres, 
                            iou_thres=self.iou_thres, 
                            max_det=self.max_det
                        )
        
        bbox = []
        conf_scores = []
        classes = []

        for det in pred:  # per image
            im0 = im0s.copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    bbox.append(xywh)
                    conf_scores.append(conf)
                    classes.append(cls)
        return bbox, conf_scores, classes