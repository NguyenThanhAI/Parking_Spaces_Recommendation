import os
import time
import numpy as np
import cv2
import tensorflow as tf

from distutils.version import StrictVersion

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                       image_width):
    """Transforms the box masks back to full image masks.
    Embeds masks in bounding boxes of larger masks whose shapes correspond to
    image shape.
    Args:
      box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
      boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
             corners. Row i contains [ymin, xmin, ymax, xmax] of the box
             corresponding to mask i. Note that the box corners are in
             normalized coordinates.
      image_height: Image height. The output mask will have the same height as
                    the image height.
      image_width: Image width. The output mask will have the same width as the
                   image width.
    Returns:
      A tf.float32 tensor of size [num_masks, image_height, image_width].
    """
    # TODO(rathodv): Make this a public function.
    def reframe_box_masks_to_image_masks_default():
        """The default function when there are more than 0 box masks."""
        def transform_boxes_relative_to_boxes(boxes, reference_boxes):
          boxes = tf.reshape(boxes, [-1, 2, 2])
          min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
          max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
          transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
          return tf.reshape(transformed_boxes, [-1, 4])

        box_masks_expanded = tf.expand_dims(box_masks, axis=3)
        num_boxes = tf.shape(box_masks_expanded)[0]
        unit_boxes = tf.concat(
            [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
        reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
        return tf.image.crop_and_resize(
            image=box_masks_expanded,
            boxes=reverse_boxes,
            box_ind=tf.range(num_boxes),
            crop_size=[image_height, image_width],
            extrapolation_value=0.0)
    image_masks = tf.cond(
        tf.shape(box_masks)[0] > 0,
        reframe_box_masks_to_image_masks_default,
        lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
    return tf.squeeze(image_masks, axis=3)


def run_inference_for_single_image(image, graph, threshold=0.5):
    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            ops = tf.get_default_graph().get_operations()

            all_tensor_names = {output.name for op in ops for output in op.outputs}

            tensor_dict = {}

            for key in ["num_detections", "detection_boxes", "detection_scores", "detection_classes",
                        "detection_masks"]:
                tensor_name = key + ":0"
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            if "detection_masks" in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict["detection_boxes"], [0])
                detection_masks = tf.squeeze(tensor_dict["detection_masks"], [0])

                real_num_detection = tf.cast(tensor_dict["num_detections"][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, threshold), tf.uint8)
                tensor_dict["detection_masks"] = tf.expand_dims(detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")

            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            output_dict["num_detections"] = int(output_dict["num_detections"][0])
            output_dict["detection_classes"] = output_dict["detection_classes"][0].astype(np.uint8)
            output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
            output_dict["detection_scores"] = output_dict["detection_scores"][0]

            if "detection_masks" in output_dict:
                output_dict["detection_masks"] = output_dict["detection_masks"][0]

            return output_dict

VIDEO_PATH = r"H:\04_前沢SA_上\201909\20190914\カメラ1\2019-09-14_17-00-00.mp4"

PATH_TO_FROZEN_GRAPH = "frozen_inference_graph.pb"

graph = tf.Graph()

with graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name="")

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print(fps, length, height, width)

frame_index = 50000

if frame_index >=0 and frame_index < length:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

ret, img = cap.read()
rgb_img = img[:, :, ::-1]

start = time.time()
output_dict = run_inference_for_single_image(image=rgb_img, graph=graph)
end = time.time()

print(round(end - start, 3))

for box, class_id, score in zip(list(output_dict["detection_boxes"]), list(output_dict["detection_classes"]), list(output_dict["detection_scores"])):
    if np.equal(box, np.zeros_like(box)).all() and class_id not in [3, 6, 8]:
        continue
    y_min, x_min, y_max, x_max = box
    cv2.rectangle(img, (int(x_min * width), int(y_min * height)), (int(x_max * width), int(y_max * height)), (255, 255, 0), 1)
    cv2.putText(img, str(class_id) + " " + str(score), (int(x_min * width) + 10, int(y_min * height) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 0))

cv2.imshow("", img)
cv2.waitKey(0)
