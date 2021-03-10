import pixellib
from pixellib.instance import instance_segmentation

segment_image = instance_segmentation(infer_speed = "fast")

segment_image.load_model("mask_rcnn_coco.h5")
# segment_image.load_model("deeplabv3_xception65_ade20k.h5")


target_classes = segment_image.select_target_classes(person=True)
src_img = "./detection/frames/frame90.jpg"

# segmask, output = segment_image.segmentImage(src_img+".png", 
#                            show_bboxes = True, 
#                            segment_target_classes=target_classes, 
#                            extract_segmented_objects= False, 
#                            save_extracted_objects=False,
#                            output_image_name = "{}-person-only.jpg".format(src_img)
#                            )
import time

tic = time.perf_counter()
segmask, output = segment_image.segmentFrame(src_img, 
                           show_bboxes = True, 
                           segment_target_classes=target_classes, 
                           extract_segmented_objects= False, 
                           save_extracted_objects=False,
                           output_image_name = "./detection/output/{}-person-only.jpg".format(src_img)
                           )
toc = time.perf_counter()

print("SEG MASKS:", segmask['extracted_objects'])
print("OUTPUT:", output)
print("OBJECTS_SEG:", len(segmask['extracted_objects']))
print("OBJECTS_OUTPUT:", len(output))
print("BBOX COORDINATES:", len(segmask['rois']))
print("CLASS IDS:", len(segmask['class_ids']))
print(f"TIME: {toc - tic:0.4f} seconds")