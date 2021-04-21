import tensorflow as tf
import cv2


def read_img(path, size):
    img = tf.image.decode_image(open(path, "rb").read(), channels=3)
    pixels = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)
    img = tf.image.resize(img, (size, size)) / 255
    img = tf.reshape(img, (1, size, size, 3))
    return (pixels, img)


def add_bboxes(pixels, boxes, scores, classes):
    (h, w, _) = pixels.shape
    for i in range(len(boxes)):
        (_x, _y, _w, _h) = boxes[i]
        (x1, y1, x2, y2) = (_x - _w / 2, _y - _h / 2, _x + _w / 2, _y + _h / 2)
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        pixels = cv2.rectangle(pixels, p1, p2, (255, 0, 0), 2)
        label = classes[i] + ": " + str(round(scores[i], 2))
        t_size = cv2.getTextSize(label, 0, 0.5, thickness=int(0.6 * (h + w) / 600) // 2)[0]
        cv2.rectangle(pixels, p1, (p1[0] + t_size[0], p1[1] - t_size[1] - 3), (255, 0, 0), -1)
        cv2.putText(
            pixels,
            label,
            (p1[0], p1[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            int(0.6 * (h + w) / 600) // 2,
            lineType=cv2.LINE_AA,
        )
    return pixels


def draw_img(pixels):
    cv2.imshow("Image", pixels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_img(filename, pixels):
    cv2.imwrite(filename, pixels)
