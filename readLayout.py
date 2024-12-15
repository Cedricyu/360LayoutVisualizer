import json
import cv2
import numpy as np

# Function to read the layout JSON file
def read_layout(file_path):
    with open(file_path, 'r') as file:
        layout = json.load(file)
    return layout

# Function to extract wall heights
def extract_wall_heights(layout):
    # Default room height from layoutHeight
    default_wall_height = layout.get('layoutHeight', None)

    # Extract cameraHeight and cameraCeilingHeight if available
    camera_height = layout.get('cameraHeight', 1.6)
    camera_ceiling_height = layout.get('cameraCeilingHeight', None)

    if not camera_ceiling_height:
        # Derive cameraCeilingHeight if not explicitly given
        camera_ceiling_height = default_wall_height - camera_height

    wall_heights = []
    for wall in layout.get('layoutWalls', {}).get('walls', []):
        # Check if wall-specific height (width) is available
        wall_height = wall.get('width', default_wall_height)
        wall_heights.append({
            "wall_id": wall["id"],
            "wall_height": wall_height
        })

    return wall_heights

# Function to draw a single wall given two points and specific wall properties
def draw_wall(img, p1, p2, wall_height, camera_height, img_height, color=(0, 0, 255)):
    """
    Draw a wall on a 2D image given two points (p1, p2) and the actual wall height.

    :param img: The image to draw on
    :param p1: First point of the wall (2D, e.g., (x1, y1))
    :param p2: Second point of the wall (2D, e.g., (x2, y2))
    :param wall_height: Total height of the wall in meters
    :param camera_height: Height of the camera in meters
    :param img_height: Total image height in pixels
    :param color: Color of the wall in BGR
    """
    # Calculate the scale factor to convert real-world height to image height
    scale_factor = img_height / (2 * camera_height)  # Scale based on camera height

    # Calculate the base and top points of the wall
    p1_base = (p1[0], p1[1] + int(camera_height * scale_factor))
    p2_base = (p2[0], p2[1] + int(camera_height * scale_factor))
    p1_top = (p1[0], p1[1] - int((wall_height - camera_height) * scale_factor))
    p2_top = (p2[0], p2[1] - int((wall_height - camera_height) * scale_factor))

    # Draw the base of the wall
    cv2.line(img, p1_base, p2_base, color, 2)

    # Draw the vertical edges of the wall
    cv2.line(img, p1_base, p1_top, color, 2)
    cv2.line(img, p2_base, p2_top, color, 2)

    # Draw the top of the wall
    cv2.line(img, p1_top, p2_top, color, 2)

# Function to mark the camera position on the image
def mark_camera_position(img, camera_coords, img_width, img_height):
    """
    Mark the camera position on the image.

    :param img: The image to draw on
    :param camera_coords: The (u, v) coordinates of the camera in normalized values
    :param img_width: Image width in pixels
    :param img_height: Image height in pixels
    """
    # Convert normalized (u, v) to pixel coordinates
    u, v = camera_coords
    camera_x = int(u * img_width)
    camera_y = int(v * img_height)

    # Draw the camera position as a red circle
    cv2.circle(img, (camera_x, camera_y), 10, (0, 0, 255), -1)  # Red dot
    cv2.putText(img, "Camera", (camera_x + 15, camera_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Function to draw the entire layout on an image
def draw_layout_with_camera(layout, input_image_path, output_image_path):
    # Load the panoramic image
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {input_image_path}")

    # Get image dimensions
    img_height, img_width, _ = img.shape

    # Extract layout points, walls, and camera info
    points = layout['layoutPoints']['points']
    walls = layout['layoutWalls']['walls']
    wall_heights = extract_wall_heights(layout)
    camera_height = layout['cameraHeight']
    camera_coords = (0.5, 0.5)  # Assume camera is at the center of the panoramic image

    # Map the 2D (u,v) coordinates to image pixel coordinates
    coords_2d = []
    for point in points:
        u, v = point['coords']
        x_pixel = int(u * img_width)  # Scale u to image width
        y_pixel = int(v * img_height)  # Scale v to image height
        coords_2d.append((x_pixel, y_pixel))

    # Draw the layout walls with specific heights
    for wall, wall_info in zip(walls, wall_heights):
        p1_idx, p2_idx = wall['pointsIdx']
        wall_height = wall_info['wall_height']  # Extract wall height from wall_heights
        p1 = coords_2d[p1_idx]
        p2 = coords_2d[p2_idx]
        draw_wall(img, p1, p2, wall_height, camera_height, img_height)

    # Mark the camera position
    mark_camera_position(img, camera_coords, img_width, img_height)

    # Save the output image
    cv2.imwrite(output_image_path, img)
    print(f"Layout visualization with specific wall heights saved to {output_image_path}")

# Main function to execute the script
if __name__ == "__main__":
    # Paths to your layout JSON file and panoramic image
    layout_file = "001_res.json"  # Replace with your layout JSON file path
    input_image = "001.png"  # Replace with your panoramic image file path
    output_image = "layout_with_specific_wall_heights.png"

    # Read the layout
    layout_data = read_layout(layout_file)

    # Draw the layout with camera on the panoramic image
    draw_layout_with_camera(layout_data, input_image, output_image)
