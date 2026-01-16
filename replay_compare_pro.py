import gymnasium as gym
import numpy as np
import time
import pygame  # 导入用于检测窗口事件

def replay_system():
    # 1. 数据加载
    try:
        ql_hist = np.load("history_ql.npy", allow_pickle=True).item()
        ql_qt = np.load("q_table_ql.npy")
        sarsa_hist = np.load("history_sarsa.npy", allow_pickle=True).item()
        sarsa_qt = np.load("q_table_sarsa.npy")
        
        data = {
            "1": {"name": "Q-Learning", "qt": ql_qt, "hist": ql_hist},
            "2": {"name": "SARSA", "qt": sarsa_qt, "hist": sarsa_hist}
        }
    except FileNotFoundError:
        print("错误：未找到数据文件，请先运行训练脚本。")
        return

    # 2. 交互循环
    while True:
        print("\n" + "="*30)
        print("1: Q-Learning | 2: SARSA | Q: 退出程序")
        choice = input("请选择算法: ").strip().lower()
        
        if choice == 'q': break
        if choice not in data: continue
        
        selected = data[choice]
        val = input(f"[{selected['name']}] 输入轮次(0-499) 或 'best' (按Esc可中止回放): ").strip()
        
        # 3. 启动回放环境
        env = gym.make("CliffWalking-v1", render_mode="human")
        state, _ = env.reset()
        
        # 准备动作序列
        if val.lower() == 'best':
            # 实时计算最优路径
            actions = []
            temp_state = state
            for _ in range(100): # 防止死循环
                a = np.argmax(selected['qt'][temp_state])
                actions.append(a)
                # 模拟逻辑位置更新（不改变环境）
                row, col = temp_state // 12, temp_state % 12
                if a == 0: row = max(0, row-1)
                elif a == 1: col = min(11, col+1)
                elif a == 2: row = min(3, row+1)
                elif a == 3: col = max(0, col-1)
                temp_state = row * 12 + col
                if temp_state == 47: break
        else:
            try:
                actions = selected['hist'][int(val)]
            except:
                print("无效轮次"); env.close(); continue

        # 4. 执行渲染回放
        print("正在播放... 点击窗口红叉或按 Esc 键退出回放")
        interrupted = False
        for a in actions:
            # --- 关键：监听 Pygame 事件 ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # 点击红叉
                    interrupted = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: # 按下 Esc
                        interrupted = True
            
            if interrupted: break
            
            env.step(a)
            env.render()
            time.sleep(0.05)
            
        env.close() # 每一轮结束都彻底关闭并清理窗口
        if interrupted:
            print("回放已由用户中止。")
        else:
            print("回放完成。")

    print("程序已安全退出。")

if __name__ == "__main__":
    replay_system()