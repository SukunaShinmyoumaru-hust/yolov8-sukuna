from ultralytics import YOLO
 
if __name__ == '__main__':
    # Load a model
    model = YOLO(r'\deploy\yolov8n.pt')  # pretrained YOLOv8n model
    model.predict(
        source=r'\deploy\output_video.mp4',
        save=False,  # save predict results
        imgsz=320,  # (int) size of input images as integer or w,h
        conf=0.25,  # object confidence threshold for detection (default 0.25 predict, 0.001 val)
        iou=0.7,  # # intersection over union (IoU) threshold for NMS
        show=True,  # show results if possible
        project='',  # (str, optional) project name
        name='',  # (str, optional) experiment name, results saved to 'project/name' directory
        save_txt=False,  # save results as .txt file
        save_conf=True,  # save results with confidence scores
        save_crop=False,  # save cropped images with results
        show_labels=True,  # show object labels in plots
        show_conf=True,  # show object confidence scores in plots
        vid_stride=1,  # video frame-rate stride
        line_width=1,  # bounding box thickness (pixels)
        visualize=False,  # visualize model features
        augment=False,  # apply image augmentation to prediction sources
        agnostic_nms=False,  # class-agnostic NMS
        retina_masks=False,  # use high-resolution segmentation masks
        boxes=True,  # Show boxes in segmentation predictions
    )