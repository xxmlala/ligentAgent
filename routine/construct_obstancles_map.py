import matplotlib.pyplot as plt
import numpy as np
import time
import math

def construct_origin_rectangle(H, W, area0, area1, instances_list, except_instances, start_point, end_point):
    unit_size = 0.01
    matrix_rows = int(H / unit_size)
    matrix_cols = int(W / unit_size)

    origin_rectangle = np.zeros((matrix_rows, matrix_cols))

    def is_valid_area(x, z):
        return area0[0] <= x <= area0[1] and area0[2] <= z <= area0[3] \
               or area1[0] <= x <= area1[1] and area1[2] <= z <= area1[3]

    def is_valid_position(x, z):
        return is_valid_area(x * unit_size - W/2, z * unit_size - H/2) and origin_rectangle[z][x] != 2

    for instance in instances_list:
        is_excepted = False
        for except_instance in except_instances:
            if except_instance in instance['prefab']:
                is_excepted = True
                break
        if is_excepted:
            continue
        size_x = int(instance["size"]["x"] / unit_size)
        size_z = int(instance["size"]["z"] / unit_size)
        pos_x = int((instance["position"]["x"] + W/2) / unit_size)
        pos_z = int((instance["position"]["z"] + H/2) / unit_size)

        for z in range(max(pos_z - size_z//2,0), min(pos_z + size_z//2, matrix_rows)):
            for x in range(max(pos_x - size_x//2,0), min(pos_x + size_x//2, matrix_cols)):
                if is_valid_position(x, z):
                    origin_rectangle[z][x] = 1

    for z in range(matrix_rows):
        for x in range(matrix_cols):
            if not is_valid_area(x * unit_size - W/2, z * unit_size - H/2):
                origin_rectangle[z][x] = 2

    return origin_rectangle, (int(start_point["position"]["z"]/0.01+matrix_rows/2), int(start_point["position"]["x"]//0.01+matrix_cols/2)), \
                            (int(end_point["position"]["z"]/0.01+matrix_rows/2), int(end_point["position"]["x"]/0.01+matrix_cols/2)) # transform the origin coordinate to the 2d array index


def compress_path(H, W, start_point, end_point, obstacles_map, path, readjust2env=False):
    if path is None: # It doesn't exist a path here.
        return None
    unit_size = 0.01
    # start_point = (int((start_point['position']['z']+H/2)/unit_size),  int((start_point['position']['x']+W/2)/unit_size))
    # end_point = (int((end_point['position']['z']+H/2)/unit_size),  int((end_point['position']['x']+W/2)/unit_size))
    def has_obstacle(start_point, end_point, obstacles_map):
        if start_point==end_point:
            return False
        z0, x0 = start_point
        z1, x1 = end_point
        dz = z1 - z0
        dx = x1 - x0
        distance = math.sqrt(dz**2 + dx**2)
        step_z = dz / distance
        step_x = dx / distance

        for i in range(int(distance)):
            z = z0 + step_z * i
            x = x0 + step_x * i
            if obstacles_map[int(z)][int(x)] != 0:
                return True
        return False
    
    compressed_path = [{'z':start_point[0],'x':start_point[1]}]
    current_point = start_point
    i = 0
    
    while i < len(path) - 1:
        next_point = path[i + 1]
        if not has_obstacle(current_point, next_point, obstacles_map):
            i += 1
        else:
            compressed_path.append({'z':path[i][0],'x':path[i][1]})
            current_point = path[i]
        i += 1
    
    compressed_path.append({'z':path[-1][0],'x':path[-1][1]})
    if readjust2env:
        for i in compressed_path:
            i['z'] = i['z'] * unit_size - H/2
            i['x'] = i['x'] * unit_size - W/2
    return compressed_path


def plot_origin_rectangle(origin_rectangle, start_point, end_point, path, compressed_path=None, save_path=None):
    unit_size = 0.01
    rows = len(origin_rectangle)
    cols = len(origin_rectangle[0])

    plt.imshow(origin_rectangle, cmap='binary', origin='lower', extent=[-cols/2, cols/2, -rows/2, rows/2])

    # Plot start point as green
    plt.plot(start_point["position"]["x"]/0.01, start_point["position"]["z"]/0.01, 'go', markersize=8, label='Start Point')

    # Plot end point as red
    plt.plot(end_point["position"]["x"]/0.01, end_point["position"]["z"]/0.01, 'ro', markersize=8, label='End Point')

    if path:
        Z, X = zip(*path)
        X = [i-cols/2 for i in X]
        Z = [i-rows/2 for i in Z]
        plt.plot(X, Z, label='path')
        assert compressed_path is not None
        for point in compressed_path:
            z,x = point['z'], point['x']
            z -= rows/2
            x -= cols/2
            plt.plot(x, z, 'bo', markersize=4, label='compressed path')

    plt.colorbar(ticks=[0, 1, 2])
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Origin Rectangle")
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


def visualization_obstacles_map(H, W, area0, area1, instances_list, start_point, end_point, except_instances=['Floor'], save_path = "./obstacles_map.png"):
    time_ = time.time()
    origin_rectangle,_,_ = construct_origin_rectangle(H, W, area0, area1, instances_list, except_instances)
    plot_origin_rectangle(origin_rectangle, start_point, end_point, save_path)
    print(f"Construction costs {time.time()-time_} s.")

if __name__=="__main__":
    H, W = 17.5, 17.5
    # 示例用法
    area0 = (-1.25,6.25,-8.75,1.25)
    area1 = (-3.75,3.75,-6.25,1.25)
    instances_list = [
        {"prefab": "item1", "size": {"x": 0.2, "z": 0.2}, "position": {"x": 0, "z": 0}},
        {"prefab": "item2", "size": {"x": 0.3, "z": 0.3}, "position": {"x": -0.5, "z": -0.5}},
    ]
    start_point = {"prefab": "start", "size": {"x": 0.1, "z": 0.1}, "position": {"x": -1.0, "z": -1.0}}
    end_point = {"prefab": "end", "size": {"x": 0.1, "z": 0.1}, "position": {"x": 1.0, "z": 1.0}}

    # origin_rectangle = construct_origin_rectangle(H, W, area0, area1, instances_list)

    # save_path = "./obstacles_map.png"
    # plot_origin_rectangle(origin_rectangle, start_point, end_point, save_path)
    visualization_obstacles_map(H, W, area0, area1, instances_list, start_point, end_point)