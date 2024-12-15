import os
import sys
import numpy as np
import cv2
from imageio import imread
import json
import argparse

from Viewer import Utils
from texture import print_lines_uv, calculate_uv, mark_points_on_image, render_uv_mapped_polygon
from wall import construct_walls, visualize_constructed_walls


def xyz_to_uv_mvp(xyz, mvp_matrix, img_width, img_height):
    """
    Convert 3D coordinates (x, y, z) to equirectangular image coordinates (u, v)
    using a Model-View-Projection (MVP) matrix.
    """
    # Append a 1.0 for the w-component to make the coordinates homogeneous
    ones = np.ones((xyz.shape[0], 1))
    xyz_h = np.hstack([xyz, ones])  # Nx4 array

    # Apply the MVP matrix
    transformed = xyz_h @ mvp_matrix.T  # Nx4 array

    # Perspective divide (convert from homogeneous to 3D coordinates)
    ndc = transformed[:, :3] / transformed[:,
                                           # Nx3 array (x, y, z in NDC)
                                           3][:, None]

    # Normalize to UV space
    u = (ndc[:, 0] + 1) / 2 * img_width  # Normalize x to [0, img_width]
    v = (1 - ndc[:, 1]) / 2 * img_height  # Normalize y to [0, img_height]

    return np.stack([u, v], axis=-1)


def construct_model_matrix(rotation_deg, vertical_rotation_deg, translation_y=0, flip_horizontal=False, flip_vertical=True):
    """
    Construct the model matrix with rotation, translation, and flipping.
    :param rotation_deg: Horizontal rotation in degrees.
    :param vertical_rotation_deg: Vertical rotation in degrees.
    :param translation_y: Translation along the y-axis to center the model.
    :param flip_horizontal: Apply horizontal flip if True.
    :param flip_vertical: Apply vertical flip if True.
    """
    # Convert angles to radians
    rotation_rad = np.radians(rotation_deg)
    vertical_rotation_rad = np.radians(vertical_rotation_deg)

    # Rotation around the y-axis (horizontal rotation)
    rotation_y = np.array([
        [np.cos(rotation_rad), 0, np.sin(rotation_rad), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation_rad), 0, np.cos(rotation_rad), 0],
        [0, 0, 0, 1]
    ])

    # Rotation around the x-axis (vertical rotation)
    rotation_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(vertical_rotation_rad), -np.sin(vertical_rotation_rad), 0],
        [0, np.sin(vertical_rotation_rad), np.cos(vertical_rotation_rad), 0],
        [0, 0, 0, 1]
    ])

    # Translation matrix for the y-axis
    translation_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, translation_y],  # Apply translation along the y-axis
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Flip matrix
    flip_matrix = np.eye(4)  # Start with an identity matrix
    if flip_horizontal:
        flip_matrix[0, 0] = -1  # Flip x-axis
    if flip_vertical:
        flip_matrix[1, 1] = -1  # Flip y-axis

    # Combine transformations: Flip * Translation * Rotation
    model_matrix = flip_matrix @ translation_matrix @ rotation_y @ rotation_x

    return model_matrix


def construct_view_matrix(distance=-10.0):
    """
    Construct the view matrix with a translation along the z-axis to move the camera back.
    :param distance: Distance to move the camera back (negative value for moving backward).
    """
    view_matrix = np.eye(4)
    view_matrix[2, 3] = distance  # Move the camera along the z-axis
    return view_matrix


def construct_projection_matrix(fov_deg, aspect_ratio, near, far):
    fov_rad = np.radians(fov_deg)
    f = 1 / np.tan(fov_rad / 2)

    projection_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])
    return projection_matrix


def construct_mvp_matrix(rotation_deg, vertical_rotation_deg, fov_deg, aspect_ratio, near, far):
    model_matrix = construct_model_matrix(rotation_deg, vertical_rotation_deg)
    view_matrix = construct_view_matrix()
    projection_matrix = construct_projection_matrix(
        fov_deg, aspect_ratio, near, far)

    # Combine: MVP = Projection * View * Model
    mvp_matrix = projection_matrix @ view_matrix @ model_matrix
    return mvp_matrix


def transform_lines_to_uv(lines_3d, mvp_matrix, img_width, img_height):
    """
    Transform 3D lines into 2D UV coordinates using the MVP matrix.
    :param lines_3d: List of 3D line segments, where each segment is [[x1, y1, z1], [x2, y2, z2]].
    :param mvp_matrix: The MVP matrix for transformation.
    :param img_width: Width of the output image.
    :param img_height: Height of the output image.
    :return: List of 2D UV coordinates for each line.
    """
    uv_lines = []
    for line in lines_3d:
        start_3d, end_3d = line
        # Transform the start and end points
        start_uv = xyz_to_uv_mvp(
            np.array([start_3d]), mvp_matrix, img_width, img_height)[0]
        end_uv = xyz_to_uv_mvp(
            np.array([end_3d]), mvp_matrix, img_width, img_height)[0]
        # Append the UV line
        uv_lines.append([start_uv, end_uv])
    return uv_lines


def draw_lines_on_image(image, uv_lines, color=(0, 255, 0), thickness=2):
    """
    Draw lines on the image using precomputed UV coordinates.
    :param image: The image to draw on.
    :param uv_lines: List of 2D UV coordinates for each line.
    :param color: Line color (default: green).
    :param thickness: Line thickness.
    :return: Image with lines drawn.
    """
    for line in uv_lines:
        start_uv, end_uv = line

        # Convert to integer pixel coordinates
        start_pixel = tuple(map(int, start_uv))
        end_pixel = tuple(map(int, end_uv))

        # Draw the line on the image
        cv2.line(image, start_pixel, end_pixel, color, thickness)

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='360 Layout Visualizer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', type=str, required=True,
                        help='The panorama path')
    parser.add_argument('--json', type=str, required=True,
                        help='The output json path')
    parser.add_argument('--rotation', type=float, default=0.0,
                        help='Rotation angle in degrees')
    parser.add_argument('--vertical_rotation', type=float, default=0.0,
                        help='Vertical rotation angle in degrees (tilt up/down)')
    parser.add_argument('--fov', type=float, default=45.0,
                        help='Field of view in degrees')
    parser.add_argument('--output', type=str, default='output_layout_with_mvp.jpg',
                        help='Path to save the output image')
    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.img):
        sys.exit(f"Error: Panorama image '{args.img}' not found.")
    if not os.path.exists(args.json):
        sys.exit(f"Error: JSON layout file '{args.json}' not found.")

    # Load panorama image
    img = imread(args.img, pilmode='RGB')
    if img.shape[2] == 4:  # Remove alpha channel if present
        img = img[:, :, :3]
    img_height, img_width = img.shape[:2]

    # Load layout
    with open(args.json, 'r') as f:
        layout = json.load(f)
    layout = Utils.OldFormat2Mine(layout)
    wall_num, wall_xz, [lines_ceiling, lines_wall,
                        lines_floor], mesh = Utils.Label2Mesh(layout)

    # Construct MVP matrix
    aspect_ratio = img_width / img_height
    mvp_matrix = construct_mvp_matrix(
        args.rotation, args.vertical_rotation, args.fov, aspect_ratio, 1., 50.0)

    # Draw ceiling, wall, and floor lines
    # Reshape into pairs of points
    lines_ceiling = lines_ceiling.reshape(-1, 2, 3)  # Ensure pairs of points
    lines_wall = lines_wall.reshape(-1, 2, 3)
    lines_floor = lines_floor.reshape(-1, 2, 3)
    line_texture_uvs = print_lines_uv(lines_floor, img_width, img_height)

    output_image = mark_points_on_image(
        img.copy(), line_texture_uvs, img_width, img_height)

    cv2.imwrite("marked_points.jpg", output_image)

    floor_uv_lines = transform_lines_to_uv(
        lines_floor, mvp_matrix, img_width=1024, img_height=512)

    new_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    # print("output lines :", floor_uv_lines)
    # print("texture uv:", line_texture_uvs)

    new_img = render_uv_mapped_polygon(
        floor_uv_lines, line_texture_uvs, img, new_img, img_width, img_height, alpha=1.0)

    walls = construct_walls(lines_wall, lines_ceiling, lines_floor)
    print("walls No :", len(walls))
    visualize_constructed_walls(walls)
    for idx, wall_lines in enumerate(walls): 
        wall_uv_lines = transform_lines_to_uv(
            wall_lines, mvp_matrix, img_width=1024, img_height=512)
        wall_texture_uvs = print_lines_uv(wall_lines, img_width, img_height, debug = False)
        output_image = mark_points_on_image(
            img.copy(), wall_texture_uvs, img_width, img_height)
        cv2.imwrite(f"marked_points_wall_{idx}.jpg", output_image)
        new_img = render_uv_mapped_polygon(
            wall_uv_lines, wall_texture_uvs, img, new_img, img_width, img_height, alpha=1.0)
    # Save the output image
    output_path = args.output
    cv2.imwrite(output_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
    print(f"Output image with lines saved to {output_path}")
