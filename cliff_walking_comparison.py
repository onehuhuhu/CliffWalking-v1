import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Agent 类定义 (保持不变) ---
class Agent:
    def __init__(self, action_shape, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((48, action_shape)) # 4x12=48个状态
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, 4) # 探索 (0:上, 1:右, 2:下, 3:左)
        else:
            return np.argmax(self.q_table[state]) # 开发

# --- 核心更新函数 (保持不变) ---
# Q-Learning 更新逻辑
def update_q_learning(q_table, s, a, r, s_next, lr, gamma):
    td_target = r + gamma * np.max(q_table[s_next])
    q_table[s, a] += lr * (td_target - q_table[s, a])

# SARSA 更新逻辑
def update_sarsa(q_table, s, a, r, s_next, a_next, lr, gamma):
    td_target = r + gamma * q_table[s_next, a_next]
    q_table[s, a] += lr * (td_target - q_table[s, a])

# --- 训练函数 ---
def train(env, agent_type="q-learning", total_episodes=500, epsilon_decay=0.995, min_epsilon=0.01):
    agent = Agent(env.action_space.n)
    rewards_per_episode = []
    
    for episode in range(total_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        # SARSA 特有的：在 S_t+1 状态下预先选好 A_t+1
        if agent_type == "sarsa":
            action = agent.choose_action(state) 
            
        while not done and not truncated:
            if agent_type == "q-learning":
                action = agent.choose_action(state) # Q-L 在循环内选择当前动作
                
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

            if agent_type == "q-learning":
                update_q_learning(agent.q_table, state, action, reward, next_state, agent.lr, agent.gamma)
            elif agent_type == "sarsa":
                next_action = agent.choose_action(next_state) # SARSA 选出下一个动作
                update_sarsa(agent.q_table, state, action, reward, next_state, next_action, agent.lr, agent.gamma)
                action = next_action # SARSA 的 S_t+1 和 A_t+1 准备好给下一轮循环
            
            state = next_state
        
        rewards_per_episode.append(episode_reward)
        
        # 衰减 epsilon
        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{total_episodes}, Agent: {agent_type}, Epsilon: {agent.epsilon:.2f}, Reward: {rewards_per_episode[-1]:.2f}")

    return agent, rewards_per_episode

# --- 路径可视化函数 ---
def plot_path(q_table, env_map, title):
    rows, cols = env_map.shape
    path = []
    
    # 找到起点 (S) 和终点 (G)
    start_pos = (rows - 1, 0)
    goal_pos = (rows - 1, cols - 1)
    
    current_state = env_map[start_pos] # 获取起始状态的编码
    path.append(start_pos)

    # 绘制路径
    done = False
    while not done:
        action = np.argmax(q_table[current_state]) # 总是选择 Q 值最大的动作
        
        # 模拟一步，但只用于获取下一个状态的坐标，不实际执行环境动作
        # action_map: {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
        current_row, current_col = path[-1]
        
        if action == 0: current_row -= 1 # Up
        elif action == 1: current_col += 1 # Right
        elif action == 2: current_row += 1 # Down
        elif action == 3: current_col -= 1 # Left

        # 边界检查
        current_row = max(0, min(rows - 1, current_row))
        current_col = max(0, min(cols - 1, current_col))

        path.append((current_row, current_col))
        
        # 如果到达终点，或者陷入循环 (为了避免无限循环)
        if (current_row, current_col) == goal_pos:
            done = True
        
        # 获取新状态的编码 (Gymnasium 环境的编码)
        current_state = env_map[current_row, current_col]
        
        if len(path) > rows * cols * 2: # 防止无限循环
            print("Warning: Path seems to be infinite loop, stopping early.")
            break
            
    # 绘制地图
    fig, ax = plt.subplots(figsize=(cols, rows))
    
    # 定义颜色映射
    cmap_data = np.array([[1.0, 1.0, 1.0, 1.0],  # 白色 (Path)
                          [0.8, 0.8, 0.8, 1.0],  # 灰色 (Safe Area)
                          [0.0, 0.0, 0.0, 1.0],  # 黑色 (Cliff)
                          [0.0, 0.5, 0.0, 1.0],  # 深绿 (Start)
                          [0.8, 0.0, 0.0, 1.0]]) # 深红 (Goal)
    cmap = ListedColormap(cmap_data)

    # 创建一个用于绘图的 grid，标记不同区域
    plot_grid = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            if env_map[r,c] == 36: # 起点 (3,0)
                plot_grid[r,c] = 3 # 绿色
            elif env_map[r,c] == 47: # 终点 (3,11)
                plot_grid[r,c] = 4 # 红色
            elif r == rows - 1 and c > 0 and c < cols - 1: # 悬崖 (3, 1~10)
                plot_grid[r,c] = 2 # 黑色
            else:
                plot_grid[r,c] = 1 # 灰色

    ax.imshow(plot_grid, cmap=cmap, origin='upper', extent=[0, cols, rows, 0])

    # 绘制路径
    path_x = [p[1] + 0.5 for p in path]
    path_y = [p[0] + 0.5 for p in path]
    ax.plot(path_x, path_y, color='blue', linewidth=2, marker='o', markersize=4)

    # 绘制网格线
    ax.set_xticks(np.arange(0, cols, 1))
    ax.set_yticks(np.arange(0, rows, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='major', color='gray', linestyle='-', linewidth=1)
    
    # 标记 S 和 G
    ax.text(start_pos[1] + 0.5, start_pos[0] + 0.5, 'S', ha='center', va='center', color='white', fontsize=16, weight='bold')
    ax.text(goal_pos[1] + 0.5, goal_pos[0] + 0.5, 'G', ha='center', va='center', color='white', fontsize=16, weight='bold')

    plt.title(title)
    plt.tight_layout()
    plt.show()

# --- 主运行部分 ---
if __name__ == "__main__":
    env = gym.make("CliffWalking-v1")

    # CliffWalking-v1 的状态空间是 0 到 47，对应 4x12 的格子
    # 状态到 (row, col) 的映射: state_idx // 12, state_idx % 12
    # 为了绘图方便，我们创建一个 map，存储每个 (row, col) 对应的 state 编号
    env_map = np.arange(48).reshape(4, 12)

    # --- 训练 Q-Learning ---
    print("--- Training Q-Learning ---")
    q_agent, q_rewards = train(env, agent_type="q-learning", total_episodes=500, epsilon_decay=0.995, min_epsilon=0.01)

    # --- 训练 SARSA ---
    print("\n--- Training SARSA ---")
    sarsa_agent, sarsa_rewards = train(env, agent_type="sarsa", total_episodes=500, epsilon_decay=0.995, min_epsilon=0.01)

    env.close()

    # --- 绘制奖励曲线 ---
    plt.figure(figsize=(12, 6))
    plt.plot(q_rewards, label="Q-Learning Rewards")
    plt.plot(sarsa_rewards, label="SARSA Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward per Episode for Q-Learning vs SARSA (Cliff Walking)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 绘制 Q-Learning 学习到的最优路径 ---
    plot_path(q_agent.q_table, env_map, "Q-Learning Learned Optimal Path")

    # --- 绘制 SARSA 学习到的最优路径 ---
    plot_path(sarsa_agent.q_table, env_map, "SARSA Learned Optimal Path")