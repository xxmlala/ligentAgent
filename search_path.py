import ligent
from ligent.server.server import set_object_counts
import json
from routine.construct_obstancles_map import compress_path, construct_origin_rectangle, plot_origin_rectangle
from routine.Search_2D.Astar import AStar
from routine.Search_2D.plotting import Plotting
import cProfile
import argparse
import time
def get_areas_info(path='./last_scene.json'):
    with open(path, 'r') as file:
        data = json.load(file)
    return data['H'], data['W'], data['area0'], data['area1']

def post_process(game_states):
    with open("../LIGENT/ligent/server/addressables.json", 'r') as file:
        items_meta_info = json.load(file)['prefabs']
    for instance in game_states['instances']:
        for item in items_meta_info:
            if not item['name'] in instance['prefab']:
                continue
            instance.update({'size':item['size']})
            break
    return game_states

def get_start_end_point(game_states, goal_name:str):
    start_point = game_states['playmate']
    if goal_name == 'player':
        end_point = game_states['player']
    else:
        instances_list = [i['prefab'] for i in game_states['instances']]
        goal_object_ids = []
        for i,instance in enumerate(instances_list):
            # if instance==goal_object: #Pumpkin(Clone) == Pumpkin
            if goal_name in instance:
                goal_object_ids.append(i)
        assert len(goal_object_ids)==1,f"len(collect_object_idx) is {len(goal_object_ids)}"
        goal_object_id = goal_object_ids[0]
        end_point = game_states['instances'][goal_object_id]
    return start_point, end_point
    
def get_path_to_goal(env, env_info, goal_name, episode_id, readjust2env=False, save_path='./obstacles_map'):
    action_noop = {
        "move_right": 0,
        "move_forward": 0,
        "look_yaw": 0.0,
        "look_pitch": 0.0,
        "jump": False,
        "grab": False,
        "speak": "",
        }
    H, W, area0, area1 = get_areas_info()
    if env is not None:
        _, _, _, info = env.step(**action_noop)
    else:
        assert env_info is not None
        info = env_info
    game_states = post_process(info["game_states"])
    start_point, end_point = get_start_end_point(game_states, goal_name)
    origin_rectangle, s_start, s_goal = construct_origin_rectangle(H, W, area0, area1, game_states['instances'], ['Floor','Catpet', goal_name], start_point, end_point)
    # plot_origin_rectangle(origin_rectangle, start_point, end_point, path=None, save_path=f"{save_path}/episode_{episode_id:04d}.png")
    astar = AStar(s_start, s_goal, "euclidean", origin_rectangle)
    path, _ = astar.searching()
    compressed_path = compress_path(H, W, s_start, s_goal, origin_rectangle, path, readjust2env)
    plot_origin_rectangle(origin_rectangle, start_point, end_point, path, compress_path(H, W, s_start, s_goal, origin_rectangle, path), save_path=f"{save_path}/episode_{episode_id:04d}_{goal_name}.png")
    return compressed_path


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--episode_num', default=5, type=int, help='episdoe num')
    parser.add_argument('-g', '--goal_name', default='player', type=str, help='goal name')
    args = parser.parse_args()
    set_object_counts({"ChristmasTree_01": 1, "Pumpkin": 1, "Watermelon_01": 1})
    env = ligent.Environment(path="C:/Users/19355/Desktop/drlProject/ligent-windows/06061943_win_224/LIGENT.exe")
    action_noop = {
        "move_right": 0,
        "move_forward": 0,
        "look_yaw": 0.0,
        "look_pitch": 0.0,
        "jump": False,
        "grab": False,
        "speak": "",
        }
    episodes_num = args.episode_num
    goal_name = args.goal_name
    try:
        for i in range(episodes_num):
            print(f"Episode {i}...")
            env.reset()
            # path = get_path_to_goal(env, goal_name, i)
            t_start = time.time()
            get_path_to_goal(env, None, "player", i)
            print(f"path to player costs {time.time()-t_start} s!")
            t_start = time.time()
            get_path_to_goal(env, None, "ChristmasTree_01", i)
            print(f"path to player costs {time.time()-t_start} s!")
            t_start = time.time()
            get_path_to_goal(env, None, "Watermelon_01", i)
            print(f"path to player costs {time.time()-t_start} s!")
            # get_path_to_goal(env, None, "ChristmasTree_01", i)
    except Exception as e:
        raise e
        # print("Exception:", e)
    finally:
        env.close()
    exit()

# set_object_counts({"ChristmasTree_01": 1, "Pumpkin": 2})
# env = ligent.Environment(path="C:/Users/19355/Desktop/drlProject/ligent-windows/06061943_win_224/LIGENT.exe")

# action_noop = {
#         "move_right": 0,
#         "move_forward": 0,
#         "look_yaw": 0.0,
#         "look_pitch": 0.0,
#         "jump": False,
#         "grab": False,
#         "speak": "",
#         }
# action_forward = {
#         "move_right": 0,
#         "move_forward": 1,
#         "look_yaw": 0.0,
#         "look_pitch": 0.0,
#         "jump": False,
#         "grab": False,
#         "speak": "",
#         }
# # env.reset()
# # H, W, area0, area1 = get_areas_info()
# # _, _, _, info = env.step(**action_noop)
# episodes_num = 10
# try:
#     for i in range(episodes_num):
#         print(f"Episode {i}...")
#         env.reset()
#         H, W, area0, area1 = get_areas_info()
#         _, _, _, info = env.step(**action_noop)
#         game_states = post_process(info["game_states"])
#         # post_process(game_states)
#         start_point = game_states['player']
#         end_point = game_states['playmate']
#         origin_rectangle, s_start, s_goal = construct_origin_rectangle(H, W, area0, area1, game_states['instances'], ['Floor','Catpet'], start_point, end_point)
        
#         plot_origin_rectangle(origin_rectangle, start_point, end_point, path=None, save_path=f"./obstacles_map/episode_{i:03d}.png")
#         astar = AStar(s_start, s_goal, "euclidean", origin_rectangle)
#         # plot = Plotting(s_start, s_goal, astar.get_env())
#         # astar.searching()
#         path, visited = astar.searching()
#         plot_origin_rectangle(origin_rectangle, start_point, end_point, path, compressed_path=compress_path(H, W, s_start, s_goal, origin_rectangle, path), save_path=f"./obstacles_map/episode_{i:03d}_Astar.png")
#         # plot.animation(path, visited, "A*")  # animation
#         # visualization_obstacles_map(H, W, area0, area1, game_states['instances'], start_point, end_point, except_instances=['Floor','Catpet'], save_path=f"./obstacles_map/step_{i//10:03d}.png")
#         # _,_,_, info = env.step(**action_forward)
#         # break
# except Exception as e:
#     raise e
#     # print("Exception:", e)
# finally:
#     env.close()
