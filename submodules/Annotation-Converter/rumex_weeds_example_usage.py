from annotation_converter.AnnotationConverter import AnnotationConverter
import glob
import os
import cv2
import random
import argparse
import matplotlib.pyplot as plt

colors = {"rumex_obtusifolius": (255, 215, 0),
          "rumex_crispus": (255, 69, 0)}


def project_mask_on_img(mask, img):
    out = cv2.addWeighted(mask, 0.5, img, 1.0, 0)
    return out

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--rumex_weeds_path", type=str)
    args = argparse.parse_args()

    annotation_file = f"{args.rumex_weeds_path}/annotations.xml"
    # Load image + mask
    img_files = glob.glob(f"{args.rumex_weeds_path}/2021*/*/imgs/*.png")

    for i in range(5):
        img_file = random.choice(img_files)
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Bounding Boxes
        annotation_file = f"{os.path.dirname(img_file)}/../annotations.xml"
        print(annotation_file)
        annotation = AnnotationConverter.read_cvat_by_id(annotation_file, os.path.basename(img_file))
        bbs = annotation.get_bounding_boxes()
        for bb in bbs:
            label = bb.get_label()
            cv2.rectangle(img, (int(bb.get_x()), int(bb.get_y())), (int(bb.get_x() + bb.get_width()), int(bb.get_y() + bb.get_height())), colors[label], 10)
        

        # segmentation mask
        annotation_file = f"{os.path.dirname(img_file)}/../annotations_seg.xml"
        annotation = AnnotationConverter.read_cvat_by_id(annotation_file, os.path.basename(img_file))
        for label in ["rumex_obtusifolius", "rumex_crispus"]:
            seg_mask = AnnotationConverter.get_mask(annotation, [label], img.shape[0], img.shape[1], colors[label])
            img = project_mask_on_img(seg_mask, img)


        plt.imshow(img)
        plt.show()

