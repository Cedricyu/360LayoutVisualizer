import cv2
import numpy as np


def calculate_uv(x, y, z):
    """
    Calculate UV coordinates (longitude and latitude) from 3D coordinates.

    Parameters:
    - x, y, z: 3D coordinates of the point.

    Returns:
    - (u, v): UV coordinates in the range [0, 1].
    """
    # Compute the vector length (magnitude) for normalization
    norm_xyz = np.sqrt(x**2 + y**2 + z**2)  # Total magnitude
    norm_xz = np.sqrt(x**2 + z**2)          # XZ plane magnitude

    # Compute longitude (u)
    lon = (np.arctan2(x, z) / np.pi + 1) * 0.5

    # Compute latitude (v)
    lat = (np.arcsin(y / norm_xyz) / (0.5 * np.pi) + 1) * 0.5

    return lon, lat


def print_lines_uv(lines, img_width, img_height, debug=True):
    """
    Compute and return UV coordinates for all lines, and optionally print them.

    Parameters:
    - lines: A numpy array of shape (N, 2, 3), where each line is defined by two 3D points.
    - img_width: Width of the image (used for UV scaling, if necessary).
    - img_height: Height of the image (used for UV scaling, if necessary).

    Returns:
    - line_uvs: A list of UV coordinates for all lines in the format:
      [
        [(u_start, v_start), (u_end, v_end)],
        ...
      ]
    """
    line_uvs = []  # To store UV coordinates for all lines
    if debug:
        print("UV Coordinates for All Lines:")
    for i, line in enumerate(lines):
        # Extract start and end points of the line
        start, end = line

        # Calculate UV for the start and end points
        u_start, v_start = calculate_uv(*start)
        u_end, v_end = calculate_uv(*end)

        # Store the UV coordinates for the current line
        line_uvs.append([(u_start, v_start), (u_end, v_end)])

        # Print UV coordinates (optional)
        if debug:
            print(f"Line {
                i + 1}: Start UV = ({u_start:.4f}, {v_start:.4f}), End UV = ({u_end:.4f}, {v_end:.4f})")

    return line_uvs


def mark_points_on_image(image, lines, img_width, img_height):
    """
    Mark UV points on the image for visualization.

    Parameters:
    - image: The original image (numpy array).
    - lines: Array of lines with UV coordinates [(u_start, v_start), (u_end, v_end)].
    - img_width: Width of the image.
    - img_height: Height of the image.

    Returns:
    - Image with marked points and lines.
    """
    for i, line in enumerate(lines):
        u_start, v_start = line[0]
        u_end, v_end = line[1]

        # Convert UV to pixel coordinates
        x_start, y_start = int(u_start * img_width), int(v_start * img_height)
        x_end, y_end = int(u_end * img_width), int(v_end * img_height)

        # Draw start and end points
        cv2.circle(image, (x_start, y_start), 5,
                   (0, 255, 0), -1)  # Green for start
        cv2.circle(image, (x_end, y_end), 5, (0, 0, 255), -1)    # Red for end

        # Draw line connecting the points
        cv2.line(image, (x_start, y_start), (x_end, y_end),
                 (255, 0, 0), 2)  # Blue line

        # Label the line
        cv2.putText(image, f"Line {i+1}", (x_start, y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def sort_polygon_points(points):
    center = np.mean(points, axis=0)
    sorted_points = sorted(points, key=lambda p: np.arctan2(
        p[1] - center[1], p[0] - center[0]))
    return np.array(sorted_points, dtype=np.float32)


def render_uv_mapped_polygon(lines_uv, texture_uv, origin_image, output_image, img_width, img_height, alpha=1.0, fill_color=(0, 255, 0)):
    """
    Render a filled polygon mapped from texture UVs to the output image.

    Parameters:
    - lines_uv: List of output image coordinates in the format [[(x1, y1), (x2, y2)], ...].
    - texture_uv: List of normalized UV coordinates in the format [[(u1, v1), (u2, v2)], ...].
    - origin_image: The origin image (numpy array).
    - img_width: Width of the output image.
    - img_height: Height of the output image.
    - alpha: Alpha value for blending (default: 1.0).
    - fill_color: RGB color to fill the polygon (default: green).

    Returns:
    - output_image: The resulting image with the polygon rendered.
    """
    # Create an empty output image and mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Flatten and normalize lines_uv
    points_out = []
    for line in lines_uv:
        for point in line:
            if list(point) not in points_out:  # Check for duplicates
                points_out.append(list(point))

    points_out = sort_polygon_points(points_out)

    # Normalize x coordinates
    points_out = np.array(points_out, dtype=np.float32)
    points_out[:, 0] /= img_width
    points_out[:, 1] /= img_height  # Normalize y coordinates

    # Flatten and scale texture_uv
    points_tex = np.array(
        [point for line in texture_uv for point in line], dtype=np.float32)
    points_tex[:, 0] *= origin_image.shape[1]  # Scale u to origin image width
    points_tex[:, 1] *= origin_image.shape[0]  # Scale v to origin image height

    # Convert normalized lines_uv back to pixel coordinates for output mask
    polygon_out = (points_out * [img_width, img_height]).astype(np.int32)
    # print("No :", len(polygon_out))

    # cv2.fillPoly(fill_layer, [polygon_out], fill_color)

    # Blend the fill color with the existing output image
    # Mask to track filled regions
    mask = cv2.fillPoly(mask, [polygon_out], 255)
    # Compute the bounding box of the polygon to reduce computation
    x_min, y_min = np.min(polygon_out, axis=0)
    x_max, y_max = np.max(polygon_out, axis=0)

    # Iterate through all pixels inside the polygon
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if mask[y, x] > 0:  # Pixel is inside the polygon
                    if mask[y, x] > 0:  # Pixel is inside the polygon
                        # Convert pixel to normalized coordinates
                        u_out = x / img_width
                        v_out = y / img_height

                        # Compute barycentric coordinates
                        def barycentric_coords(p, a, b, c):
                            denom = (b[1] - c[1]) * (a[0] - c[0]) + \
                                (c[0] - b[0]) * (a[1] - c[1])
                            w1 = ((b[1] - c[1]) * (p[0] - c[0]) +
                                (c[0] - b[0]) * (p[1] - c[1])) / denom
                            w2 = ((c[1] - a[1]) * (p[0] - c[0]) +
                                (a[0] - c[0]) * (p[1] - c[1])) / denom
                            w3 = 1 - w1 - w2
                            return w1, w2, w3

                        # Triangulate the polygon and find the triangle containing (u_out, v_out)
                        for i in range(len(points_out) - 2):
                            a, b, c = points_out[0], points_out[i +
                                                                1], points_out[i + 2]
                            if cv2.pointPolygonTest(np.array([a, b, c], dtype=np.float32), (u_out, v_out), False) >= 0:
                                # Compute barycentric coordinates
                                w1, w2, w3 = barycentric_coords(
                                    [u_out, v_out], a, b, c)

                                # Interpolate the texture UV
                                tex_u = w1 * \
                                    points_tex[0][0] + w2 * points_tex[i +
                                                                    1][0] + w3 * points_tex[i + 2][0]
                                tex_v = w1 * \
                                    points_tex[0][1] + w2 * points_tex[i +
                                                                    1][1] + w3 * points_tex[i + 2][1]

                                # Map to the texture
                                tex_x = int(tex_u)
                                tex_y = int(tex_v)
                                if 0 <= tex_x < origin_image.shape[1] and 0 <= tex_y < origin_image.shape[0]:
                                    color = origin_image[tex_y, tex_x]

                                    # Alpha blending
                                    output_image[y, x] = (
                                        alpha * color + (1 - alpha) * output_image[y, x]).astype(np.uint8)

    return output_image
