import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt

# --- (1) 环境定义 Env Class - 与您提供的基线环境基本相同，但会添加一些改进的说明 ---
# 为了完整性，这里再次提供环境定义。在实际应用中，您可以根据需求进一步优化。
class Env:
    def __init__(self, num_firms=4, initial_inventory=10, initial_backlog=0,
                 max_order=10, max_inventory=50, lead_time=1,
                 holding_cost=0.5, backlog_cost=1.0, order_cost_multiplier=0.1,
                 demand_type='poisson', poisson_lambda=5,
                 seasonal_amplitude=3, seasonal_period=10, trend_rate=0.1,
                 historical_window=3, # 新增: 历史数据窗口
                 inventory_variance_penalty=0.01): # 新增: 库存波动惩罚
        
        self.num_firms = num_firms
        self.initial_inventory = initial_inventory
        self.initial_backlog = initial_backlog
        self.max_order = max_order
        self.max_inventory = max_inventory
        self.lead_time = lead_time
        self.holding_cost = holding_cost
        self.backlog_cost = backlog_cost
        self.order_cost_multiplier = order_cost_multiplier
        
        # 需求模式参数
        self.demand_type = demand_type
        self.poisson_lambda = poisson_lambda
        self.seasonal_amplitude = seasonal_amplitude
        self.seasonal_period = seasonal_period
        self.trend_rate = trend_rate
        self.current_step = 0 # 用于跟踪时间步，以便生成趋势和季节性需求

        # 新增奖励函数参数
        self.inventory_variance_penalty = inventory_variance_penalty
        self.previous_inventories = [] # 用于计算库存方差

        self.firms = []
        for i in range(num_firms):
            self.firms.append({
                'inventory': initial_inventory,
                'backlog': initial_backlog,
                'incoming_orders': deque([0] * lead_time, maxlen=lead_time), # 消费者需求或上游订单
                'outgoing_shipments': deque([0] * lead_time, maxlen=lead_time), # 发货给下游
                'placed_orders': deque([0] * lead_time, maxlen=lead_time), # 下给上游的订单
                'incoming_delivery': deque([0] * lead_time, maxlen=lead_time), # 上游发来的货
                'current_order_to_upstream': 0, # 当前周期下给上游的订单
                'current_demand': 0, # 当前周期面临的需求 (来自消费者或下游)
                'satisfied_demand': 0, # 当前周期满足的需求
                'lost_sales': 0, # 当前周期损失的销售
                'profit': 0,
                'history': deque(maxlen=historical_window) # 存储历史 (orders, satisfied_demand, inventory)
            })
            # 初始历史填充
            self.firms[i]['history'].append(np.array([initial_inventory, 0, initial_inventory])) # Initial state (inv, satisfied, inv)

        self.reset() # 调用 reset 来初始化状态

    def reset(self):
        for firm in self.firms:
            firm['inventory'] = self.initial_inventory
            firm['backlog'] = self.initial_backlog
            firm['incoming_orders'] = deque([0] * self.lead_time, maxlen=self.lead_time)
            firm['outgoing_shipments'] = deque([0] * self.lead_time, maxlen=self.lead_time)
            firm['placed_orders'] = deque([0] * self.lead_time, maxlen=self.lead_time)
            firm['incoming_delivery'] = deque([0] * self.lead_time, maxlen=self.lead_time)
            firm['current_order_to_upstream'] = 0
            firm['current_demand'] = 0
            firm['satisfied_demand'] = 0
            firm['lost_sales'] = 0
            firm['profit'] = 0
            firm['history'] = deque(maxlen=self.firms[0]['history'].maxlen) # 重置历史
            firm['history'].append(np.array([self.initial_inventory, 0, self.initial_inventory]))

        self.current_step = 0
        self.previous_inventories = [] # 重置库存历史
        
        # 初始观察值 (对于 RL agent，通常是第一个 firm)
        return self._get_observation(0)

    def _generate_demand(self):
        if self.demand_type == 'poisson':
            return np.random.poisson(self.poisson_lambda)
        elif self.demand_type == 'seasonal':
            # 季节性需求: 基础泊松 + 正弦波
            base_lambda = self.poisson_lambda
            season_factor = self.seasonal_amplitude * np.sin(2 * np.pi * self.current_step / self.seasonal_period)
            return max(0, int(np.random.poisson(base_lambda + season_factor)))
        elif self.demand_type == 'trend':
            # 趋势需求: 基础泊松 + 线性趋势
            base_lambda = self.poisson_lambda
            trend_factor = self.trend_rate * self.current_step
            return max(0, int(np.random.poisson(base_lambda + trend_factor)))
        elif self.demand_type == 'seasonal_trend':
            # 季节性与趋势结合
            base_lambda = self.poisson_lambda
            season_factor = self.seasonal_amplitude * np.sin(2 * np.pi * self.current_step / self.seasonal_period)
            trend_factor = self.trend_rate * self.current_step
            return max(0, int(np.random.poisson(base_lambda + season_factor + trend_factor)))
        else:
            raise ValueError("Unsupported demand type")

    def _get_observation(self, firm_idx):
        firm = self.firms[firm_idx]
        
        # 将历史数据扁平化
        history_flat = np.array(list(firm['history'])).flatten()
        
        # 订单，满足的需求，库存，以及历史数据
        # 确保 outstanding_orders 也是一个单值或扁平化的序列
        # 考虑加入当前即将收到的货和即将发出的货 (in-transit inventory)
        
        # outstanding_orders: 已经下单但尚未收到的订单总和
        outstanding_orders = sum(firm['placed_orders']) # placed_orders 队列中是过去已下订单
        
        # incoming_delivery: 即将到货的量
        # outgoing_shipments: 即将发出的量 (给下游，作为下游的incoming_orders)
        
        # 状态空间包含：
        # [当前库存, 当前积压订单, 当前incoming_orders队列,
        #  当前placed_orders队列, 历史数据扁平化, outstanding_orders]
        
        # incoming_orders 队列表示来自下游的需求，或对于零售商是消费者需求
        # 队列中的值是历史的，最新的值会在step中更新
        
        # 为了简化和保持维度一致性，我们取队列的当前值和其和
        # (或者直接将整个队列作为状态的一部分，但需要固定队列长度)
        
        # 我们重新定义一下状态，使其更具代表性
        # current_inventory: 当前库存
        # current_backlog: 当前积压订单
        # latest_customer_order: 最新的顾客订单 (对于firm 0 是实际需求, 对于其他firm是下游订单)
        # latest_supplier_delivery: 最新的供应商交付 (对于firm 0 没有上游交付, 其他firm有)
        # outstanding_orders_to_supplier: 已经下给供应商但未收到的订单总和
        
        state_elements = [
            firm['inventory'],
            firm['backlog'],
            firm['incoming_orders'][-1] if firm['incoming_orders'] else 0, # 最近一期的下游订单/需求
            firm['incoming_delivery'][-1] if firm['incoming_delivery'] else 0, # 最近一期的到货
            outstanding_orders,
            self.current_step # 加入当前时间步作为状态的一部分，帮助模型识别季节和趋势
        ]
        
        # 额外加入历史库存、历史满足需求、历史订单（RL agent 自己下给上游的）
        # 假设 history deque 存储的是 [inventory, satisfied_demand, placed_order_by_agent]
        history_for_state = []
        for h_step in firm['history']:
            history_for_state.extend([h_step[0], h_step[1], h_step[2]])
            
        state_elements.extend(history_for_state)
        
        return np.array(state_elements, dtype=np.float32)


    def step(self, actions):
        self.current_step += 1
        rewards = [0] * self.num_firms
        done = False # 简单起见，暂不设置终止条件

        # 生成最下游公司的需求
        customer_demand = self._generate_demand()
        self.firms[0]['current_demand'] = customer_demand

        # 向上游传递订单并处理发货和收货
        # Firm 0 (零售商) 接收消费者需求，将订单传给 Firm 1 (批发商)
        # Firm 1 接收 Firm 0 的订单，将订单传给 Firm 2 (分销商)
        # ...
        # Firm N-1 接收 Firm N-2 的订单，将订单传给外部供应商 (假定永远有货)

        # 1. 记录本期RL agent 的订单 (RL agent 假定为 firm_idx=0)
        # 此时 actions 应该是一个列表，RL agent 的动作是 actions[0]
        # 或者在训练函数中只传入 RL agent 的 action
        
        # 为了让 env.step 能够处理所有 firm 的 action，我们假设 actions 是一个列表
        # actions = [order_firm0, order_firm1, ..., order_firm_N-1]
        
        # 2. 处理订单传递和库存更新
        # 从最下游公司开始处理
        for i in range(self.num_firms):
            firm = self.firms[i]
            
            # 接收上游到货 (滞后 lead_time)
            # 对于 Firm 0，其上游是 Firm 1。Firm 1的发货在 lead_time 之前被记录在 Firm 0 的 incoming_delivery 队列里
            # 对于 Firm N-1，假定外部供应商总是及时发货，且数量等于 Firm N-1 的订单量
            if i == self.num_firms - 1: # 最上游的厂商，没有上游订单滞后，货直接到
                 firm['incoming_delivery'].append(firm['current_order_to_upstream'])
            else:
                 # 从队列中取出 lead_time 周期前的发货
                 firm['incoming_delivery'].append(self.firms[i+1]['outgoing_shipments'].popleft())

            # 实际收到货，增加库存
            firm['inventory'] += firm['incoming_delivery'][-1]
            
            # 处理积压订单和满足需求
            # 先处理积压订单
            demand_to_fulfill = firm['backlog']
            firm['satisfied_demand'] = 0
            firm['lost_sales'] = 0

            if firm['inventory'] >= demand_to_fulfill:
                firm['inventory'] -= demand_to_fulfill
                firm['satisfied_demand'] += demand_to_fulfill
                firm['backlog'] = 0
            else:
                firm['satisfied_demand'] += firm['inventory']
                firm['backlog'] -= firm['inventory']
                firm['inventory'] = 0
            
            # 处理本期需求
            current_demand_for_firm = firm['current_demand'] if i == 0 else firm['incoming_orders'][-1]
            demand_to_fulfill = current_demand_for_firm

            if firm['inventory'] >= demand_to_fulfill:
                firm['inventory'] -= demand_to_fulfill
                firm['satisfied_demand'] += demand_to_fulfill
            else:
                firm['satisfied_demand'] += firm['inventory']
                firm['lost_sales'] += (demand_to_fulfill - firm['inventory'])
                firm['backlog'] += (demand_to_fulfill - firm['inventory'])
                firm['inventory'] = 0

            # 存储历史库存用于计算方差
            if i == 0: # 只对RL agent 计算库存方差
                 self.previous_inventories.append(firm['inventory'])
            
            # 记录发货给下游 (或消费者)
            firm['outgoing_shipments'].append(firm['satisfied_demand'])

            # RL agent 动作 (订单给上游)
            order_to_upstream = int(actions[i])
            order_to_upstream = max(0, min(order_to_upstream, self.max_order)) # 限制订单量
            
            firm['current_order_to_upstream'] = order_to_upstream
            firm['placed_orders'].append(order_to_upstream) # 记录下给上游的订单
            
            # 向上游传递订单 (下一周期成为上游的 incoming_orders)
            if i < self.num_firms - 1:
                self.firms[i+1]['incoming_orders'].append(order_to_upstream)
            else:
                # 最上游公司，没有上游，需求直接消失
                pass # 或者可以模拟外部供应商的某种行为

        # 3. 计算奖励和更新状态
        for i in range(self.num_firms):
            firm = self.firms[i]
            
            # 成本
            holding_cost = self.holding_cost * firm['inventory']
            backlog_cost = self.backlog_cost * firm['backlog']
            order_cost = self.order_cost_multiplier * firm['current_order_to_upstream']
            
            # 销售收入 (只考虑零售商的销售，或每个环节的“销售”都是向下游发货)
            # 为了简化，我们假设 RL 代理的“销售”是其 satisfied_demand
            # 其他公司的“销售”是其 outgoing_shipments
            
            revenue = firm['satisfied_demand'] # 这里简化为 satisfied_demand 作为“销售收入”

            # 基本奖励: 销售收入 - 各种成本
            reward = revenue - holding_cost - backlog_cost - order_cost
            
            # 添加库存波动惩罚 (只对RL agent)
            if i == 0 and len(self.previous_inventories) > 1:
                inv_std = np.std(self.previous_inventories)
                reward -= self.inventory_variance_penalty * inv_std
            
            rewards[i] = reward
            firm['profit'] += reward # 累积利润

            # 更新历史
            # history 存储 [inventory, satisfied_demand, placed_order_by_agent]
            # 为了状态维度不变，placed_order_by_agent 应该是 RL agent 的订单
            # 如果是其他 firm，这个值是其下给上游的订单
            firm['history'].append(np.array([firm['inventory'], firm['satisfied_demand'], firm['current_order_to_upstream']]))

        # 返回 RL agent (firm 0) 的观察值、奖励、是否终止等信息
        next_observation = self._get_observation(0)
        
        return next_observation, rewards[0], done, {}


# --- (2) QNetwork 定义 ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # 增加网络深度和宽度
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size) # 输出层到动作空间

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- (3) ReplayBuffer 定义 ---
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# --- (4) DQNAgent 定义 - Double DQN 核心修改 ---
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device) # 目标网络
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # DDQN Modification:
        # Action selection from local network:
        action_indices = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        # Q-value evaluation from target network:
        Q_targets_next = self.qnetwork_target(next_states).gather(1, action_indices)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local.gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)                     

    def soft_update(self, local_model, target_model, tau=1e-3):
        # Soft update model parameters.
        # target_model = tau*local_model + (1.0-tau)*target_model
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# --- (5) 训练与测试函数 ---

# 新增：经典库存管理策略
class ClassicAgent:
    def __init__(self, policy_type='sS', s=5, S=20, base_stock_level=15, max_order=10):
        self.policy_type = policy_type
        self.s = s  # Reorder point for (s,S)
        self.S = S  # Order-up-to level for (s,S) and Base Stock
        self.base_stock_level = base_stock_level # Base Stock Policy Level
        self.max_order = max_order

    def act(self, state, env_firm_data):
        inventory = env_firm_data['inventory']
        backlog = env_firm_data['backlog']
        incoming_delivery = env_firm_data['incoming_delivery'] # 队列的最后一个是本周期收到的货
        placed_orders = env_firm_data['placed_orders'] # 队列存储的是历史下给上游的订单

        # 将在途库存考虑在内 (通常是placed_orders队列的总和)
        in_transit = sum(placed_orders) # 假设placed_orders是未来会到货的
        
        # inventory_position = inventory + in_transit - backlog
        # 简化：只考虑当前库存和积压，不考虑在途，或者将 in_transit 纳入 'S' 的考量
        # 这里为了简化，我们只使用当前库存和积压来决定
        
        # 对于啤酒游戏，通常订单只在 lead_time 后到货。
        # inventory_position 应该 = current_inventory + orders_in_transit - backlog
        # 这里的 env_firm_data['placed_orders'] 存储的是之前下的订单，其总和是在途库存
        
        # 修正 inventory_position 的计算
        # current_inventory 是当前周期处理完需求后的库存
        # placed_orders 队列里的订单，是已经发出但未收到的货。
        # 最新的订单是 firm['current_order_to_upstream']，这个订单在 step() 结尾才被加入 placed_orders 队列
        # 所以这里的 placed_orders 应该是指尚未到货的旧订单
        
        # 我们用一个更简单的定义：考虑库存和积压
        
        order_quantity = 0

        if self.policy_type == 'sS':
            if inventory <= self.s:
                order_quantity = self.S - inventory + backlog # 考虑积压，使库存达到 S
        elif self.policy_type == 'BaseStock':
            # Base Stock Policy: Order up to base_stock_level
            # 考虑净库存 (Net Inventory = On-hand Inventory - Backlog)
            net_inventory = inventory - backlog
            order_quantity = self.base_stock_level - net_inventory
        elif self.policy_type == 'Random':
             order_quantity = np.random.randint(0, self.max_order + 1)
        else:
            raise ValueError("Unsupported Classic Policy Type")
        
        order_quantity = max(0, min(order_quantity, self.max_order)) # 限制订单量
        return order_quantity


def train_dqn(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    
    # 记录 RL agent 每次的利润、库存和订单
    rl_agent_profits = []
    rl_agent_inventories = []
    rl_agent_orders = []
    rl_agent_satisfied_demands = []

    # 其他 firms 的代理 (固定策略)
    # 可以尝试不同的策略组合：例如，都用随机，或都用 (s,S)，或混合
    other_firms_agents = []
    # 假设 firm 1, 2, 3 都使用 Base Stock 策略
    for i in range(1, env.num_firms):
        # 调整这些参数以适应不同层级的公司
        other_firms_agents.append(ClassicAgent(policy_type='BaseStock', base_stock_level=15 + i*5, max_order=env.max_order))
        # other_firms_agents.append(ClassicAgent(policy_type='Random', max_order=env.max_order))


    for i_episode in range(1, n_episodes + 1):
        state = env.reset() # Reset environment
        score = 0
        
        episode_firm_profits = [0] * env.num_firms
        episode_firm_inventories = [[] for _ in range(env.num_firms)]
        episode_firm_orders = [[] for _ in range(env.num_firms)]
        episode_firm_satisfied_demands = [[] for _ in range(env.num_firms)]

        for t in range(max_t):
            # RL agent (firm 0) 选择动作
            action_rl = agent.act(state, eps)
            
            # 其他 firms 选择动作
            all_actions = [action_rl]
            for i in range(1, env.num_firms):
                firm_data = env.firms[i]
                action_other = other_firms_agents[i-1].act(state=None, env_firm_data=firm_data) # ClassicAgent 不需要完整的RL state
                all_actions.append(action_other)

            # 环境执行所有动作
            next_state, rewards, done, _ = env.step(all_actions)
            
            # RL agent 学习
            agent.step(state, action_rl, rewards, next_state, done)
            
            state = next_state
            score += rewards

            # 记录RL agent 的数据
            episode_firm_profits[0] += rewards
            episode_firm_inventories[0].append(env.firms[0]['inventory'])
            episode_firm_orders[0].append(action_rl)
            episode_firm_satisfied_demands[0].append(env.firms[0]['satisfied_demand'])
            
            # 记录其他 firms 的数据 (可选，主要关注RL agent)
            for i in range(1, env.num_firms):
                episode_firm_profits[i] += env.firms[i]['profit'] # env.firms[i]['profit'] 已经是累积的
                episode_firm_inventories[i].append(env.firms[i]['inventory'])
                episode_firm_orders[i].append(all_actions[i])
                episode_firm_satisfied_demands[i].append(env.firms[i]['satisfied_demand'])

            if done:
                break 
        
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        rl_agent_profits.append(episode_firm_profits[0])
        rl_agent_inventories.append(np.mean(episode_firm_inventories[0]) if episode_firm_inventories[0] else 0)
        rl_agent_orders.append(np.mean(episode_firm_orders[0]) if episode_firm_orders[0] else 0)
        rl_agent_satisfied_demands.append(np.mean(episode_firm_satisfied_demands[0]) if episode_firm_satisfied_demands[0] else 0)


        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        # if np.mean(scores_window) >= 200.0: # Example success condition
        #     print(f'\nEnvironment solved in {i_episode-100} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
        #     break
    return scores, {
        'profits': rl_agent_profits,
        'inventories': rl_agent_inventories,
        'orders': rl_agent_orders,
        'satisfied_demands': rl_agent_satisfied_demands
    }


def test_agent(env, agent, n_episodes=10, max_t=1000):
    test_scores = []
    test_rl_agent_profits = []
    test_rl_agent_inventories = []
    test_rl_agent_orders = []
    test_rl_agent_satisfied_demands = []
    
    # 其他 firms 的代理 (固定策略) - 与训练时保持一致
    other_firms_agents = []
    for i in range(1, env.num_firms):
        other_firms_agents.append(ClassicAgent(policy_type='BaseStock', base_stock_level=15 + i*5, max_order=env.max_order))
        # other_firms_agents.append(ClassicAgent(policy_type='Random', max_order=env.max_order))


    for i_episode in range(n_episodes):
        state = env.reset()
        score = 0
        
        episode_rl_inventory = []
        episode_rl_order = []
        episode_rl_satisfied_demand = []
        
        for t in range(max_t):
            action_rl = agent.act(state, eps=0.0) # In test, use greedy policy
            
            all_actions = [action_rl]
            for i in range(1, env.num_firms):
                firm_data = env.firms[i]
                action_other = other_firms_agents[i-1].act(state=None, env_firm_data=firm_data)
                all_actions.append(action_other)

            next_state, rewards, done, _ = env.step(all_actions)
            
            state = next_state
            score += rewards
            
            episode_rl_inventory.append(env.firms[0]['inventory'])
            episode_rl_order.append(action_rl)
            episode_rl_satisfied_demand.append(env.firms[0]['satisfied_demand'])

            if done:
                break
        
        test_scores.append(score)
        test_rl_agent_profits.append(score)
        test_rl_agent_inventories.append(np.mean(episode_rl_inventory) if episode_rl_inventory else 0)
        test_rl_agent_orders.append(np.mean(episode_rl_order) if episode_rl_order else 0)
        test_rl_agent_satisfied_demands.append(np.mean(episode_rl_satisfied_demand) if episode_rl_satisfied_demand else 0)

    print(f"\n--- Test Results (Average over {n_episodes} episodes) ---")
    print(f"Average Score: {np.mean(test_scores):.2f}")
    print(f"Average Inventory: {np.mean(test_rl_agent_inventories):.2f}")
    print(f"Average Orders: {np.mean(test_rl_agent_orders):.2f}")
    print(f"Average Satisfied Demand: {np.mean(test_rl_agent_satisfied_demands):.2f}")
    
    return {
        'scores': test_scores,
        'profits': test_rl_agent_profits,
        'inventories': test_rl_agent_inventories,
        'orders': test_rl_agent_orders,
        'satisfied_demands': test_rl_agent_satisfied_demands
    }


def plot_results(training_scores, training_data, test_data, title="DQN Agent Performance"):
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))
    fig.suptitle(title)

    # 1. Training Scores
    axs[0].plot(training_scores)
    axs[0].set_title('Training Average Score per Episode')
    axs[0].set_ylabel('Score')
    axs[0].set_xlabel('Episode #')
    axs[0].grid(True)

    # 2. RL Agent Profit per Episode (Training)
    axs[1].plot(training_data['profits'])
    axs[1].set_title('RL Agent Profit per Episode (Training)')
    axs[1].set_ylabel('Profit')
    axs[1].set_xlabel('Episode #')
    axs[1].grid(True)

    # 3. RL Agent Average Inventory per Episode (Training)
    axs[2].plot(training_data['inventories'])
    axs[2].set_title('RL Agent Average Inventory per Episode (Training)')
    axs[2].set_ylabel('Average Inventory')
    axs[2].set_xlabel('Episode #')
    axs[2].grid(True)

    # 4. RL Agent Average Order Quantity per Episode (Training)
    axs[3].plot(training_data['orders'])
    axs[3].set_title('RL Agent Average Order Quantity per Episode (Training)')
    axs[3].set_ylabel('Average Order Quantity')
    axs[3].set_xlabel('Episode #')
    axs[3].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    # Test Results Visualization (simplified)
    # Could add more detailed plots for test results if needed,
    # e.g., inventory trajectory, orders, satisfied demand over time for a single test run.
    print("\nTest Results Summary:")
    print(f"Average Test Score: {np.mean(test_data['scores']):.2f}")
    print(f"Average Test Profit: {np.mean(test_data['profits']):.2f}")
    print(f"Average Test Inventory: {np.mean(test_data['inventories']):.2f}")
    print(f"Average Test Orders: {np.mean(test_data['orders']):.2f}")
    print(f"Average Test Satisfied Demand: {np.mean(test_data['satisfied_demands']):.2f}")


# --- (6) 参数设定与运行 ---
if __name__ == "__main__":
    # 环境参数
    NUM_FIRMS = 4
    INITIAL_INVENTORY = 10
    MAX_ORDER = 10
    LEAD_TIME = 1
    HOLDING_COST = 0.5
    BACKLOG_COST = 2.0 # 增加积压成本，鼓励满足需求
    ORDER_COST_MULTIPLIER = 0.05
    INVENTORY_VARIANCE_PENALTY = 0.05 # 增加库存波动惩罚

    # 需求参数 (选择一种)
    DEMAND_TYPE = 'seasonal_trend' # 'poisson', 'seasonal', 'trend', 'seasonal_trend'
    POISSON_LAMBDA = 5
    SEASONAL_AMPLITUDE = 3
    SEASONAL_PERIOD = 10 # 每10个时间步一个季节周期
    TREND_RATE = 0.1

    # 状态空间和动作空间 (需要根据 Env._get_observation 调整)
    # state_size = 5 (inv, backlog, latest_customer_order, latest_supplier_delivery, outstanding_orders_to_supplier)
    # + historical_window * 3 (inv, satisfied_demand, placed_order_by_agent)
    # + 1 (current_step)
    
    # 假设 historical_window = 3: (3历史库存 + 3历史满足需求 + 3历史订单)
    # 历史数据窗口大小，需与 Env 类中的 historical_window 保持一致
    HISTORICAL_WINDOW = 3 
    # 计算实际状态空间大小
    # [inventory, backlog, incoming_orders[-1], incoming_delivery[-1], outstanding_orders, current_step]
    # + historical_window * [inventory, satisfied_demand, current_order_to_upstream]
    STATE_SIZE = 6 + HISTORICAL_WINDOW * 3 
    ACTION_SIZE = MAX_ORDER + 1 # 0 到 max_order

    # 初始化环境
    env = Env(num_firms=NUM_FIRMS, initial_inventory=INITIAL_INVENTORY,
              max_order=MAX_ORDER, lead_time=LEAD_TIME,
              holding_cost=HOLDING_COST, backlog_cost=BACKLOG_COST,
              order_cost_multiplier=ORDER_COST_MULTIPLIER,
              demand_type=DEMAND_TYPE, poisson_lambda=POISSON_LAMBDA,
              seasonal_amplitude=SEASONAL_AMPLITUDE, seasonal_period=SEASONAL_PERIOD,
              trend_rate=TREND_RATE, historical_window=HISTORICAL_WINDOW,
              inventory_variance_penalty=INVENTORY_VARIANCE_PENALTY)

    # 初始化 Double DQN 代理
    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE, seed=0)

    # 训练代理
    print("Starting training...")
    training_scores, training_data = train_dqn(env, agent, n_episodes=5000, max_t=200) # 增加训练轮次和每轮步数
    print("Training finished.")

    # 测试代理
    print("\nStarting testing...")
    test_data = test_agent(env, agent, n_episodes=50, max_t=200) # 增加测试轮次
    print("Testing finished.")

    # 绘制结果
    plot_results(training_scores, training_data, test_data)