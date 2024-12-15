import numpy as np
import matplotlib.pyplot as plt


def remove_duplicate_lines(lines):
    """
    去掉重复的线段。

    参数:
    - lines: 原始线段列表，每个线段是 [[x1, y1, z1], [x2, y2, z2]] 格式。

    返回:
    - unique_lines: 去重后的线段列表。
    """
    unique_lines_set = set()

    for line in lines:
        # 将线段规范化为无序形式（排序后转换为元组）
        sorted_line = tuple(sorted([tuple(line[0]), tuple(line[1])]))
        unique_lines_set.add(sorted_line)

    # 将集合中的线段还原为原始格式
    unique_lines = [list(map(np.array, line)) for line in unique_lines_set]

    return unique_lines

def find_connected_line(verticals, lines):
    """
    查找与两条垂直线段连接的水平线段（floor 或 ceiling）。

    参数:
    - verticals: 两条垂直线段。
    - lines: 候选的水平线段。

    返回:
    - 连接的线段（如果找到）；否则返回 None。
    """
    for line in lines:
        start_matches = [np.allclose(line[0], v[0], atol=1e-5) or np.allclose(line[0], v[1], atol=1e-5) for v in verticals]
        end_matches = [np.allclose(line[1], v[0], atol=1e-5) or np.allclose(line[1], v[1], atol=1e-5) for v in verticals]

        # 检查起点和终点是否分别连接到不同的垂直线
        if sum(start_matches) == 1 and sum(end_matches) == 1:
            # print(f"Connected Floor Line Found: {line}")
            return line

    return None


def check_data_types(lines_wall, lines_ceiling, lines_floor):
    """
    检查 lines_wall, lines_ceiling, lines_floor 数据类型和一致性。

    参数:
    - lines_wall: 垂直线段列表。
    - lines_ceiling: 天花板水平线段列表。
    - lines_floor: 地板水平线段列表。
    """
    def check_lines(lines, name):
        print(f"Checking {name}:")
        for i, line in enumerate(lines):
            start, end = line
            print(f"  Line {i + 1}:")
            print(f"    Start: {start}, Type: {type(start)}, Shape: {start.shape}")
            print(f"    End: {end}, Type: {type(end)}, Shape: {end.shape}")
            if not (isinstance(start, np.ndarray) and isinstance(end, np.ndarray)):
                print(f"    Error: Line {i + 1} in {name} contains non-numpy data types!")
            if start.shape != (3,) or end.shape != (3,):
                print(f"    Error: Line {i + 1} in {name} has incorrect shape!")
        print()

    # 检查每个线段列表
    check_lines(lines_wall, "lines_wall")
    check_lines(lines_ceiling, "lines_ceiling")
    check_lines(lines_floor, "lines_floor")

def infer_fourth_line(known_lines):
    """
    根据三条已知线推断第四条线。

    参数:
    - known_lines: 包含三条线段的列表，每条线段由两个点构成。

    返回:
    - fourth_line: 第四条线段，由两个点构成。
    """
    # 提取所有端点
    points = []
    for line in known_lines:
        points.append(tuple(line[0]))  # 起点
        points.append(tuple(line[1]))  # 终点

    # 找到出现次数为 1 的点
    unique_points = []
    for point in points:
        if points.count(point) == 1:  # 点只出现一次
            unique_points.append(point)

    # 如果有两个唯一端点，则它们构成第四条线
    if len(unique_points) == 2:
        fourth_line = [np.array(unique_points[0], dtype=np.float32), np.array(unique_points[1], dtype=np.float32)]
        return fourth_line
    else:
        raise ValueError("无法推断第四条线，输入线段可能不正确。")

def construct_walls(lines_wall, lines_ceiling, lines_floor):
    """
    根据 lines_wall、lines_ceiling 和 lines_floor 构造墙壁，并去掉重复线段。

    参数:
    - lines_wall: 包含垂直线段的列表。
    - lines_ceiling: 包含天花板水平线段的列表。
    - lines_floor: 包含地板水平线段的列表。

    返回:
    - walls: 每个墙壁由 4 条线段组成。
    """
    # 去重线段
    # print("floor line :",(lines_floor))

    lines_wall = remove_duplicate_lines(lines_wall)
    lines_ceiling = remove_duplicate_lines(lines_ceiling)
    lines_floor = remove_duplicate_lines(lines_floor)
    # print("wall line :",(lines_wall))
    # print("ceiling line :",(lines_ceiling))
    # print("floor line :",(lines_floor))
    # check_data_types(lines_wall, lines_ceiling, lines_floor)


    walls = []

    # 遍历 lines_wall，每次取两条垂直线段
    for i in range(len(lines_wall)):
        for j in range(i + 1, len(lines_wall)):
            wall_lines = []

            # 获取 2 条垂直线段
            vertical_1 = lines_wall[i]
            vertical_2 = lines_wall[j]
            wall_lines.append(vertical_1)
            wall_lines.append(vertical_2)

            # 查找与这两条垂直线段连接的地板线（floor）
            floor = find_connected_line([vertical_1, vertical_2], lines_floor)

            # 如果找到匹配的地板线
            if floor:
                wall_lines.append(floor)
                # print("Floor Found:", wall_lines)
                wall_lines.append(infer_fourth_line(wall_lines))
                # 检查是否形成闭合墙壁（暂时只用 floor）
                if is_wall_closed(wall_lines):
                    # print("Closed Wall Found:", wall_lines)
                    walls.append(wall_lines)

    return walls


def is_wall_closed(wall_lines):
    """
    检查墙壁是否由 4 条线段组成并形成闭合。

    参数:
    - wall_lines: 墙壁的 4 条线段。

    返回:
    - True: 如果墙壁是闭合的。
    - False: 如果墙壁不是闭合的。
    """
    points = []
    for line in wall_lines:
        points.append(line[0])  # 起点
        points.append(line[1])  # 终点

    # 去掉重复的点
    unique_points = []
    for point in points:
        if not any(np.allclose(point, p, atol=1e-5) for p in unique_points):
            unique_points.append(point)

    # 检查是否是 4 个点
    return len(unique_points) == 4


def visualize_constructed_walls(walls):
    """
    可视化构造的墙壁。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制墙壁
    for idx, wall in enumerate(walls):
        for line in wall:  # 每条线段
            xs = [line[0][0], line[1][0]]
            ys = [line[0][1], line[1][1]]
            zs = [line[0][2], line[1][2]]
            ax.plot(xs, ys, zs, 'r--')  # 红色虚线表示墙壁
        # 标注墙壁编号
        x_mid = np.mean([point[0] for line in wall for point in line])
        y_mid = np.mean([point[1] for line in wall for point in line])
        z_mid = np.mean([point[2] for line in wall for point in line])
        ax.text(x_mid, y_mid, z_mid, f'W{idx}', color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
