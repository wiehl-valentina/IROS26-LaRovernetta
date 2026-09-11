import numpy as np
from genie_path_planner.costs import path_cost_with_footprint, goal_term

def evaluate(alpha, goal_weight):
    grid_size = 240
    path_straight = np.stack([np.linspace(120, 20, 100), np.linspace(120, 120, 100)], axis=1) 
    path_dodge = np.stack([np.linspace(120, 20, 100), np.linspace(120, 160, 100)], axis=1) 
    
    cost_map = np.full((240, 240), 0.5, dtype=float)
    cost_map[50:70, 110:130] = 0.65
    
    goal_rc = (0, 120)
    
    c_straight = path_cost_with_footprint(cost_map, path_straight, alpha=alpha, footprint_px=1) + goal_term(path_straight, goal_rc, goal_weight, grid_size)
    c_dodge = path_cost_with_footprint(cost_map, path_dodge, alpha=alpha, footprint_px=1) + goal_term(path_dodge, goal_rc, goal_weight, grid_size)
    
    if c_straight < c_dodge:
        print(f"Alpha {alpha} -> FAILS TO AVOID (Straight:{c_straight:.1f}, Dodge:{c_dodge:.1f})")
    else:
        print(f"Alpha {alpha} -> DODGES SUCCESSFULLY (Straight:{c_straight:.1f}, Dodge:{c_dodge:.1f})")

evaluate(2.0, 1.0)
evaluate(2.5, 1.0)
evaluate(3.0, 1.0)
