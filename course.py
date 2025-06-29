import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
class Env:
    def __init__(self, num_firms, p, h, c, initial_inventory, poisson_lambda=10, max_steps=100):
        """
        初始化供应链管理仿真环境。
        
        :param num_firms: 企业数量
        :param p: 各企业的价格列表
        :param h: 库存持有成本
        :param c: 损失销售成本
        :param initial_inventory: 每个企业的初始库存
        :param poisson_lambda: 最下游企业需求的泊松分布均值
        :param max_steps: 每个episode的最大步数
        """
        self.num_firms = num_firms
        self.p = p  # 企业的价格列表
        self.h = h  # 库存持有成本
        self.c = c  # 损失销售成本
        self.poisson_lambda = poisson_lambda  # 泊松分布的均值
        self.max_steps = max_steps  # 每个episode的最大步数
        self.initial_inventory = initial_inventory  # 初始库存
        
        # 初始化库存
        self.inventory = np.full((num_firms, 1), initial_inventory)
        # 初始化订单量
        self.orders = np.zeros((num_firms, 1))
        # 初始化已满足的需求量
        self.satisfied_demand = np.zeros((num_firms, 1))
        # 记录当前步数
        self.current_step = 0
        # 标记episode是否结束
        self.done = False

    def reset(self):
        """
        重置环境状态。
        """
        self.inventory = np.full((self.num_firms, 1), self.initial_inventory)
        self.orders = np.zeros((self.num_firms, 1))
        self.satisfied_demand = np.zeros((self.num_firms, 1))
        self.current_step = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        """
        获取每个企业的观察信息，包括订单量、满足的需求量和库存。
        每个企业的状态是独立的，包括自己观察的订单、需求和库存。
        """
        return np.concatenate((self.orders, self.satisfied_demand, self.inventory), axis=1)

    def _generate_demand(self):
        """
        根据规则生成每个企业的需求。
        最下游企业的需求遵循泊松分布，其他企业的需求等于下游企业的订单量。
        """
        demand = np.zeros((self.num_firms, 1))
        for i in range(self.num_firms):
            if i == 0:
                # 最下游企业的需求遵循泊松分布，均值为 poisson_lambda
                demand[i] = np.random.poisson(self.poisson_lambda)
            else:
                # 上游企业的需求等于下游企业的订单量
                demand[i] = self.orders[i - 1]  # d_{i+1,t} = q_{it}
        return demand

    def step(self, actions):
        """
        执行一个时间步的仿真，根据给定的行动 (每个企业的订单量) 更新环境状态。
        
        :param actions: 每个企业的订单量 (shape: (num_firms, 1))，即每个智能体的行动
        :return: next_state, reward, done
        """
        self.orders = actions  # 更新订单量
        
        # 生成各企业的需求
        self.demand = self._generate_demand()

        # 计算每个企业收到的订单量和满足的需求
        for i in range(self.num_firms):
            if i == 0:
                # 第一企业从外部需求直接得到满足
                self.satisfied_demand[i] = min(self.demand[i], self.inventory[i])
            else:
                # 后续企业的需求由上游企业订单决定
                self.satisfied_demand[i] = min(self.demand[i], self.inventory[i])
        
        # 更新库存
        for i in range(self.num_firms):
            self.inventory[i] = self.inventory[i] + self.orders[i] - self.satisfied_demand[i]
        
        # 计算每个企业的奖励: p_i * d_{it} - p_{i+1} * q_{it} - h * I_{it}
        rewards = np.zeros((self.num_firms, 1))
        loss_sales = np.zeros((self.num_firms, 1))  # 损失销售费用
        
        for i in range(self.num_firms):
            rewards[i] += self.p[i] * self.satisfied_demand[i] - (self.p[i+1] if i+1 < self.num_firms else 0) * self.orders[i] - self.h * self.inventory[i]
            
            # 损失销售计算
            if self.satisfied_demand[i] < self.demand[i]:
                loss_sales[i] = (self.demand[i] - self.satisfied_demand[i]) * self.c
        
        rewards -= loss_sales  # 总奖励扣除损失销售成本
        
        # 增加步数
        self.current_step += 1
        
        # 判断是否结束（比如达到最大步数）
        if self.current_step >= self.max_steps:
            self.done = True
        
        return self._get_observation(), rewards, self.done

# 使用示例
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 经验回放
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Double DQN Agent
class DoubleDQNAgent:
    def __init__(self, state_size, action_size, firm_id, max_order=20, buffer_size=10000, batch_size=64, 
                 gamma=0.99, learning_rate=1e-3, tau=1e-3, update_every=4):

        self.state_size = state_size
        self.action_size = action_size
        self.firm_id = firm_id
        self.max_order = max_order
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.learning_step = 0

        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def act(self, state, epsilon=0.0):
        state = torch.from_numpy(state.flatten()).float().unsqueeze(0)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()

        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy()) + 1
        else:
            return random.randint(1, self.max_order)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack([s.flatten() for s in states])).float()
        actions = torch.from_numpy(np.vstack([a-1 for a in actions])).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack([ns.flatten() for ns in next_states])).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        # Double DQN核心
        next_actions = self.q_network(next_states).detach().argmax(1).unsqueeze(1)
        Q_targets_next = self.target_network(next_states).gather(1, next_actions)

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learning_step += 1
        if self.learning_step % self.update_every == 0:
            self.soft_update()

        return loss.item()

    def soft_update(self):
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"模型已保存到 {filename}")

    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"从 {filename} 加载了模型")
            return True
        return False

def train_dqn(env, agent, num_episodes=1000, max_t=100, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    eps = eps_start
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                if firm_id == agent.firm_id:
                    firm_state = state[firm_id].reshape(1, -1)
                    action = agent.act(firm_state, eps)
                    actions[firm_id] = action
                else:
                    actions[firm_id] = np.random.randint(1, 21)
            next_state, rewards, done = env.step(actions)
            reward = rewards[agent.firm_id][0]
            agent.step(state[agent.firm_id].reshape(1, -1), actions[agent.firm_id], reward, next_state[agent.firm_id].reshape(1, -1), done)
            state = next_state
            score += reward
            if done:
                break
        eps = max(eps_end, eps_decay * eps)
        scores.append(score)
        if i_episode % 100 == 0:
            print(f'Episode {i_episode}/{num_episodes} | Average Score: {np.mean(scores[-100:]):.2f} | Epsilon: {eps:.4f}')
        if i_episode % 500 == 0:
            agent.save(f'models/double_dqn_agent_firm_{agent.firm_id}_episode_{i_episode}.pth')
    agent.save(f'models/double_dqn_agent_firm_{agent.firm_id}_final.pth')
    return scores

# 测试函数保持一致
def test_agent(env, agent, num_episodes=10):
    scores = []
    inventory_history = []
    orders_history = []
    demand_history = []
    satisfied_demand_history = []
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        score = 0
        episode_inventory = []
        episode_orders = []
        episode_demand = []
        episode_satisfied_demand = []
        for t in range(env.max_steps):
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                if firm_id == agent.firm_id:
                    firm_state = state[firm_id].reshape(1, -1)
                    action = agent.act(firm_state, epsilon=0.0)
                    actions[firm_id] = action
                else:
                    actions[firm_id] = np.random.randint(1, 21)
            next_state, rewards, done = env.step(actions)
            episode_inventory.append(env.inventory[agent.firm_id][0])
            episode_orders.append(actions[agent.firm_id][0])
            episode_demand.append(env.demand[agent.firm_id][0])
            episode_satisfied_demand.append(env.satisfied_demand[agent.firm_id][0])
            reward = rewards[agent.firm_id][0]
            score += reward
            state = next_state
            if done:
                break
        scores.append(score)
        inventory_history.append(episode_inventory)
        orders_history.append(episode_orders)
        demand_history.append(episode_demand)
        satisfied_demand_history.append(episode_satisfied_demand)
        print(f'Test Episode {i_episode}/{num_episodes} | Score: {score:.2f}')
    return scores, inventory_history, orders_history, demand_history, satisfied_demand_history


def plot_training_results(scores, window_size=100):
    """
    绘制训练结果
    
    :param scores: 每个episode的奖励
    :param window_size: 移动平均窗口大小
    """
    # 计算移动平均
    def moving_average(data, window_size):
        return [np.mean(data[max(0, i-window_size):i+1]) for i in range(len(data))]
    
    avg_scores = moving_average(scores, window_size)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores, alpha=0.3, label='Init Reward')
    plt.plot(np.arange(len(avg_scores)), avg_scores, label=f'Moving Average ({window_size} episodes)')
    plt.title(' DOUBLE-DQN Training Process Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.legend()
    plt.savefig('figures/double_dqn_training_rewards.png')
    plt.close()

def plot_test_results(scores, inventory_history, orders_history, demand_history, satisfied_demand_history):
    """
    绘制测试结果
    
    :param scores: 每个episode的奖励
    :param inventory_history: 每个episode的库存历史
    :param orders_history: 每个episode的订单历史
    :param demand_history: 每个episode的需求历史
    :param satisfied_demand_history: 每个episode的满足需求历史
    """
    # 计算平均值，用于绘图
    avg_inventory = np.mean(inventory_history, axis=0)
    avg_orders = np.mean(orders_history, axis=0)
    avg_demand = np.mean(demand_history, axis=0)
    avg_satisfied_demand = np.mean(satisfied_demand_history, axis=0)
    
    # 创建图表
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # 库存图表
    # axs[0, 0].plot(avg_inventory)
    # axs[0, 0].set_title('平均库存')
    # axs[0, 0].set_xlabel('时间步')
    # axs[0, 0].set_ylabel('库存量')
    
    # # 订单图表
    # axs[0, 1].plot(avg_orders)
    # axs[0, 1].set_title('平均订单量')
    # axs[0, 1].set_xlabel('时间步')
    # axs[0, 1].set_ylabel('订单量')
    
    # # 需求和满足需求图表
    # axs[1, 0].plot(avg_demand, label='需求')
    # axs[1, 0].plot(avg_satisfied_demand, label='满足的需求')
    # axs[1, 0].set_title('平均需求 vs 满足的需求')
    # axs[1, 0].set_xlabel('时间步')
    # axs[1, 0].set_ylabel('数量')
    # axs[1, 0].legend()
    
    # # 奖励柱状图
    # axs[1, 1].bar(range(len(scores)), scores)
    # axs[1, 1].set_title('测试episode奖励')
    # axs[1, 1].set_xlabel('Episode')
    # axs[1, 1].set_ylabel('总奖励')
        # 库存图表
    axs[0, 0].plot(avg_inventory)
    axs[0, 0].set_title('Average Inventory')
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].set_ylabel('Inventory Level')
    
    # 订单图表
    axs[0, 1].plot(avg_orders)
    axs[0, 1].set_title('Average Order Quantity')
    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].set_ylabel('Order Quantity')
    
    # 需求和满足需求图表
    axs[1, 0].plot(avg_demand, label='Demand')
    axs[1, 0].plot(avg_satisfied_demand, label='Satisfied Demand')
    axs[1, 0].set_title('Demand vs Satisfied Demand')
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('Quantity')
    axs[1, 0].legend()
    
    # 奖励柱状图
    axs[1, 1].bar(range(len(scores)), scores)
    axs[1, 1].set_title('Test Episode Rewards')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Total Reward')
    
    plt.tight_layout()
    plt.savefig('figures/double_dqn_test_results.png')
    plt.close()
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    num_firms = 3
    p = [10, 9, 8]
    h = 0.5
    c = 2
    initial_inventory = 100
    poisson_lambda = 10
    max_steps = 100
    
    env = Env(num_firms, p, h, c, initial_inventory, poisson_lambda, max_steps)
    
    firm_id = 1
    state_size = 3
    action_size = 20
    
    agent = DoubleDQNAgent(state_size=state_size, action_size=action_size, firm_id=firm_id, max_order=action_size)
    
    scores = train_dqn(env, agent, num_episodes=2000, max_t=max_steps, eps_start=1.0, eps_end=0.01, eps_decay=0.995)
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans Mono']
    plt.rcParams['axes.unicode_minus'] = False
    
    plot_training_results(scores)
    
    test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history = test_agent(env, agent, num_episodes=10)
    
    plot_test_results(test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history)