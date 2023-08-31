# PPO for ligent (in developing... 

1.只试验 A* 效果: obstacle_map generation and A* search:
```sh
python search_path.py
```
可视化结果会存在`./obstacles_map`路径下，A*搜索算法改编自 [PathPlanning](https://github.com/zhm-real/PathPlanning)

2.收集数据，参考 `collect_data.bat` 的脚本写法 (暂未详细尝试成功率，有时候没有可行路线，会在命令行输出提示)
寻路及建图结果会存在`./obstacles_map`路径下, visual_observation 会存在 `./obs_visions_<goal_name>`路径下