import numpy as np
from genie_path_planner.costs import path_cost_with_footprint, goal_term

def evaluate(alpha, goal_weight):
    grid_size = 240
    # Simulate a path of 100 points
    path_free = np.stack([np.linspace(120, 20, 100), np.linspace(120, 120, 100)], axis=1) # goes straight
    path_dodge = np.stack([np.linspace(120, 20, 100), np.linspace(120, 140, 100)], axis=1) # goes slightly right
    
    # Cost map: obstacle straight ahead from r=50 to 70, c=110 to 130
    cost_map = np.zeros((240, 240), dtype=float)
    cost_map[50:70, 110:130] = 0.6  # soft obstacle
    
    # Goal is straight ahead at (0, 120)
    goal_rc = (0, 120)
    
    c_free_raw = path_cost_with_footprint(cost_map, path_free, alpha=alpha, footprint_px=1)
    c_dodge_raw = path_cost_with_footprint(cost_map, path_dodge, alpha=alpha, footprint_px=1)
    
    gt_free = goal_term(path_free, goal_rc, goal_weight, grid_size)
    gt_dodge = goal_term(path_dodge, goal_rc, goal_weight, grid_size)
    
    c_free = c_free_raw + gt_free
    c_dodge = c_dodge_raw + gt_dodge
    
    print(f"Alpha {alpha}, GoalWeight {goal_weight}")
    print(f"  Straight (hits obst): Total={c_free:.1f} (Raw={c_free_raw:.1f} + Goal={gt_free:.1f})")
    print(f"  Dodge (avoids obst) : Total={c_dodge:.1f} (Raw={c_dodge_raw:.1f} + Goal={gt_dodge:.1f})")
    if c_free < c_dodge:
        print("  -> ROBOT HITS OBSTACLE")
    else:
        print("  -> ROBOT DODGES OBSTACLE")

evaluate(1.0, 1.0)
evaluate(3.0, 1.0)
evaluate(4.0, 1.0)
