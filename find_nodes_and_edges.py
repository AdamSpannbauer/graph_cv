import cv2
import numpy as np

input_path = 'numbered_graph.png'


def label_contour(image, contour, label, font_color=(0, 0, 255)):
    """put a text label at the center of a contour

    :param image: image to draw label on
    :param contour: contour of interest (will derive center for putting text)
    :param label: text to write on contour
    :param font_color: text font color as BGR tuple
    :return: None. modifies input image in place
    """
    moments = cv2.moments(contour)
    x_center = int(moments["m10"] / moments["m00"])
    y_center = int(moments["m01"] / moments["m00"])
    cv2.putText(image,
                f'{label}',
                (x_center, y_center),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.25,
                font_color,
                4)


# read image and convert to grayscale
im = cv2.imread(input_path)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# threshold image to more isolate edges and nodes
_, threshed_nodes = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
_, threshed_edges = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

# further isolate nodes
threshed_nodes = cv2.bitwise_xor(threshed_nodes, threshed_edges)
threshed_nodes = cv2.erode(threshed_nodes.copy(), None)
threshed_nodes = cv2.morphologyEx(threshed_nodes, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

# find contours (should be only nodes given preprocessing)
_, node_cnts, _ = cv2.findContours(threshed_nodes.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# copy input image for drawing contours on
nodes_clone = im.copy()
for (i, c) in enumerate(node_cnts):
    # draw and label contour
    cv2.drawContours(nodes_clone, [c], -1, (0, 255, 0), 2)
    label_contour(nodes_clone, c, label=i)

# further isolate edges
threshed_edges_inv = 255 - threshed_edges

# find contours (should be only nodes given preprocessing)
_, edge_cnts, _ = cv2.findContours(threshed_edges_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# copy input image for drawing contours on
edges_clone = im.copy()
valid_i = -1
for (i, c) in enumerate(edge_cnts):
    # compute the area & bounding box of contour of interest
    area = cv2.contourArea(c)

    if area > 100:
        valid_i += 1
        # draw and label contour
        cv2.drawContours(edges_clone, [c], -1, (0, 255, 0), 2)
        label_contour(edges_clone, c, label=valid_i)


# display results of preprocessing
cv2.imshow('Edges', edges_clone)
cv2.imshow('Nodes', nodes_clone)
cv2.imshow('Input Image', im)
cv2.waitKey(0)
