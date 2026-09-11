import numpy as np
from genie_path_planner.costs import path_cost_with_footprint, goal_term

def evaluate(alpha, goal_weight):
    grid_size = 240
    # Paths
    path_straight = np.stack([np.linspace(120, 20, 100), np.linspace(120, 120, 100)], axis=1) # hits obstacle
    path_dodge = np.stack([np.linspace(120, 20, 100), np.linspace(120, 160, 100)], axis=1) # dodges, but farther from goal
    
    # Cost map: uniform slope (cost 0.5) everywhere
    cost_map = np.full((240, 240), 0.5, dtype=float)
    # Obstacle on the slope straight ahead (cost 0.9)
    cost_map[50:70, 110:130] = 0.9
    
    # Goal is straight ahead
    goal_rc = (0, 120)
    
    c_straight = path_cost_with_footprint(cost_map, path_straight, alpha=alpha, footprint_px=1) + goal_term(path_straight, goal_rc, goal_weight, grid_size)
    c_dodge = path_cost_with_footprint(cost_map, path_dodge, alpha=alpha, footprint_px=1) + goal_term(path_dodge, goal_rc, goal_weight, grid_size)
    
    print(f"Alpha {alpha}, Goal {goal_weight}")
    print(f"  Straight (hits obj on slope): {c_straight:.1f}")
    print(f"  Dodge (avoids obj on slope) : {c_dodge:.1f}")
    if c_straight < c_dodge:
        print("  -> FAILS TO AVOID")
    else:
        print("  -> DODGES SUCCESSFULLY")

evaluate(1.0, 1.0)
evaluate(2.0, 1.0)
evaluate(4.0, 1.0)
