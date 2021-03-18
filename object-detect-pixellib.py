import pixellib
from pixellib.instance import instance_segmentation

segment_video = instance_segmentation(infer_speed = "rapid")
segment_video.load_model("mask_rcnn_coco.h5")
target_classes = segment_video.select_target_classes(person=True)

segment_video.process_video("./detection/mall-video-1.mp4", # input 
                            show_bboxes = True, 
                            segment_target_classes=target_classes, 
                            frames_per_second= 10000, 
                            # extract_segmented_objects=True,
                            # save_extracted_objects=True,
                            output_video_name="./detection/output_video.mp4" # output
                            )

# segment_video.process_video("./detection/mall-video-1.mp4", show_bboxes = True, frames_per_second= 0.25, output_video_name="./detection/output_video.mp4")

