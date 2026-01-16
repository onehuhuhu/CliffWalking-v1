import gymnasium as gym
import numpy as np

def run_training(algo_name):
    print(f"--- 正在训练 {algo_name} ---")
    env = gym.make("CliffWalking-v1")
    q_table = np.zeros((48, 4))
    lr, gamma, epsilon = 0.1, 0.9, 0.2
    history = {}

    for ep in range(500):
        state, _ = env.reset()
        done = False
        actions = []
        
        # SARSA 初始动作选择
        if algo_name == "sarsa":
            action = np.argmax(q_table[state]) if np.random.rand() > epsilon else env.action_space.sample()

        while not done:
            if algo_name == "q-learning":
                action = np.argmax(q_table[state]) if np.random.rand() > epsilon else env.action_space.sample()
            
            next_state, reward, term, trun, _ = env.step(action)
            done = term or trun
            actions.append(action)
            
            if algo_name == "q-learning":
                # Q-Learning 更新
                q_table[state, action] += lr * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
            else:
                # SARSA 更新
                next_act = np.argmax(q_table[next_state]) if np.random.rand() > epsilon else env.action_space.sample()
                q_table[state, action] += lr * (reward + gamma * q_table[next_state, next_act] - q_table[state, action])
                action = next_act
            
            state = next_state
        
        history[ep] = actions
        epsilon = max(0.01, epsilon * 0.99)
        if (ep + 1) % 100 == 0: print(f"Episode {ep+1}/500 完成")
        
    return q_table, history

if __name__ == "__main__":
    # 训练 Q-Learning
    qt_ql, hist_ql = run_training("q-learning")
    np.save("q_table_ql.npy", qt_ql)
    np.save("history_ql.npy", hist_ql)

    # 训练 SARSA
    qt_sarsa, hist_sarsa = run_training("sarsa")
    np.save("q_table_sarsa.npy", qt_sarsa)
    np.save("history_sarsa.npy", hist_sarsa)

    print("\n所有算法训练完毕，数据已分名保存。")