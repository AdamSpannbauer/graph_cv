import cv2
import imutils
import numpy as np

input_path = 'input.png'


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
im = imutils.resize(im, width=800)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# threshold image to more isolate edges and nodes
_, threshed_nodes = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
_, black_blobs = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

# further isolate nodes
threshed_nodes = cv2.bitwise_xor(threshed_nodes, black_blobs)
threshed_nodes = cv2.erode(threshed_nodes, None)
threshed_nodes = cv2.morphologyEx(threshed_nodes, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

# find contours (should be only nodes given preprocessing)
_, node_cnts, _ = cv2.findContours(threshed_nodes.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# copy input image for drawing contours on
nodes_clone = im.copy()
masked_edges = gray.copy()
for (i, c) in enumerate(node_cnts):
    area = cv2.contourArea(c)
    if 50 < area < 500:
        # draw and label contour
        cv2.drawContours(nodes_clone, [c], -1, (0, 255, 0), 2)
        cv2.drawContours(masked_edges, [c], -1, 255, -1)
        cv2.drawContours(masked_edges, [c], -1, 255, 5)

_, threshed_edges = cv2.threshold(masked_edges, 150, 255, cv2.THRESH_BINARY_INV)
threshed_edges = cv2.dilate(threshed_edges, None)

_, edge_cnts, _ = cv2.findContours(threshed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
edges_clone = im.copy()
cv2.drawContours(edges_clone, edge_cnts, -1, (0, 255, 0), 2)

# display results of preprocessing
cv2.imshow('Edges', imutils.resize(edges_clone, width=600))
cv2.imshow('Nodes', imutils.resize(nodes_clone, width=600))
cv2.imshow('Input Image', imutils.resize(im, width=600))
cv2.waitKey(0)
