"""
2体のAntエージェントが相撲で戦うMuJoCo環境
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class AntSumoEnv(gym.Env):
    """
    2体のAntが相撲で対戦する環境
    
    タスク：
    - 円の中で相手と戦い、相手をひっくり返すか円の外に出したら勝利
    - 勝利: +1000の報酬
    - 敗北: -1000の報酬
    - 時間経過: -1.0の報酬
    
    観測：
    - 自分の位置 (2次元)
    - 自分の関節角度 (8次元)
    - 自分の関節角速度 (8次元)
    - 自分の関節トルク (8次元)
    - 相手の位置 (2次元)
    - 相手の関節角度 (8次元)
    - 相手の関節角速度 (8次元)
    合計: 44次元
    
    行動：
    - Antの8つの関節トルク (各関節 -1 ~ 1)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, arena_radius=5.0, max_episode_steps=1000, render_mode=None):
        super().__init__()
        
        self.arena_radius = arena_radius
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.current_step = 0
        
        # MuJoCo環境の設定
        self._setup_mujoco()

        # 報酬パラメータ（典型的な相撲タスクの設計）
        # 勝利: +1000, 敗北: -1000, 時間経過: -1.0
        self.time_penalty = -1.0
        self.ant1_geom_ids = None
        self.ant2_geom_ids = None
        self.ant1_body_id = None
        self.ant2_body_id = None
          
        # 行動空間：8つの関節トルク（各エージェント）
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        # 観測空間：44次元（位置・関節角度・角速度・トルク、自他8関節ずつ）
        obs_dim = 44
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.model = None
        self.data = None
        
    def _setup_mujoco(self):
        """MuJoCoモデルのXML定義を作成"""
        # 2体のAnt用のMuJoCo XMLを定義
        xml_string = f"""
        <mujoco model="ant_sumo">
          <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
          <option integrator="RK4" timestep="0.01"/>
          
          <default>
            <joint armature="1" damping="1" limited="true"/>
            <!-- 衝突判定を有効にするため contype=1, conaffinity=1 を設定 -->
            <geom contype="1" conaffinity="1" condim="3" density="5.0" friction="1.5 0.1 0.1" margin="0.01" rgba="0.8 0.6 0.4 1"/>
          </default>
          
          <worldbody>
            <!-- 地面 -->
            <geom name="floor" pos="0 0 0" size="40 40 0.1" type="plane" rgba="0.8 0.9 0.8 1"/>
            
            <!-- 相撲の円 -->
            <geom name="arena" pos="0 0 0.01" size="{self.arena_radius} 0.05" type="cylinder" rgba="1 0.8 0.2 0.3" conaffinity="0" contype="0"/>
            
            <!-- Agent 1 (青) -->
            <body name="ant1_torso" pos="2 0 0.75">
              <geom name="ant1_torso_geom" pos="0 0 0" size="0.25" type="sphere" rgba="0.2 0.4 0.8 1"/>
              <joint armature="0" damping="0" limited="false" margin="0.01" name="ant1_root" pos="0 0 0" type="free"/>
              
              <!-- 前脚 -->
              <body name="ant1_front_left_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="ant1_aux_1" size="0.08" type="capsule" rgba="0.2 0.4 0.8 1"/>
                <body name="ant1_aux_1" pos="0.2 0.2 0">
                  <joint axis="0 0 1" name="ant1_hip_1" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                  <geom fromto="0.0 0.0 0.0 0.4 0.4 0.0" name="ant1_left_leg" size="0.08" type="capsule" rgba="0.2 0.4 0.8 1"/>
                  <body pos="0.4 0.4 0">
                    <joint axis="-1 1 0" name="ant1_ankle_1" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 0.4 0.4 0.0" name="ant1_left_ankle" size="0.08" type="capsule" rgba="0.2 0.4 0.8 1"/>
                  </body>
                </body>
              </body>
              
              <body name="ant1_front_right_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="ant1_aux_2" size="0.08" type="capsule" rgba="0.2 0.4 0.8 1"/>
                <body name="ant1_aux_2" pos="-0.2 0.2 0">
                  <joint axis="0 0 1" name="ant1_hip_2" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                  <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="ant1_right_leg" size="0.08" type="capsule" rgba="0.2 0.4 0.8 1"/>
                  <body pos="-0.4 0.4 0">
                    <joint axis="1 1 0" name="ant1_ankle_2" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="ant1_right_ankle" size="0.08" type="capsule" rgba="0.2 0.4 0.8 1"/>
                  </body>
                </body>
              </body>
              
              <!-- 後脚 -->
              <body name="ant1_back_left_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="ant1_aux_3" size="0.08" type="capsule" rgba="0.2 0.4 0.8 1"/>
                <body name="ant1_aux_3" pos="0.2 -0.2 0">
                  <joint axis="0 0 1" name="ant1_hip_3" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                  <geom fromto="0.0 0.0 0.0 0.4 -0.4 0.0" name="ant1_back_leg" size="0.08" type="capsule" rgba="0.2 0.4 0.8 1"/>
                  <body pos="0.4 -0.4 0">
                    <joint axis="-1 -1 0" name="ant1_ankle_3" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 0.4 -0.4 0.0" name="ant1_back_left_ankle" size="0.08" type="capsule" rgba="0.2 0.4 0.8 1"/>
                  </body>
                </body>
              </body>
              
              <body name="ant1_back_right_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="ant1_aux_4" size="0.08" type="capsule" rgba="0.2 0.4 0.8 1"/>
                <body name="ant1_aux_4" pos="-0.2 -0.2 0">
                  <joint axis="0 0 1" name="ant1_hip_4" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                  <geom fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" name="ant1_right_back_leg" size="0.08" type="capsule" rgba="0.2 0.4 0.8 1"/>
                  <body pos="-0.4 -0.4 0">
                    <joint axis="1 -1 0" name="ant1_ankle_4" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" name="ant1_back_right_ankle" size="0.08" type="capsule" rgba="0.2 0.4 0.8 1"/>
                  </body>
                </body>
              </body>
            </body>
            
            <!-- Agent 2 (赤) -->
            <body name="ant2_torso" pos="-2 0 0.75">
              <geom name="ant2_torso_geom" pos="0 0 0" size="0.25" type="sphere" rgba="0.8 0.2 0.2 1"/>
              <joint armature="0" damping="0" limited="false" margin="0.01" name="ant2_root" pos="0 0 0" type="free"/>
              
              <!-- 前脚 -->
              <body name="ant2_front_left_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="ant2_aux_1" size="0.08" type="capsule" rgba="0.8 0.2 0.2 1"/>
                <body name="ant2_aux_1" pos="0.2 0.2 0">
                  <joint axis="0 0 1" name="ant2_hip_1" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                  <geom fromto="0.0 0.0 0.0 0.4 0.4 0.0" name="ant2_left_leg" size="0.08" type="capsule" rgba="0.8 0.2 0.2 1"/>
                  <body pos="0.4 0.4 0">
                    <joint axis="-1 1 0" name="ant2_ankle_1" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 0.4 0.4 0.0" name="ant2_left_ankle" size="0.08" type="capsule" rgba="0.8 0.2 0.2 1"/>
                  </body>
                </body>
              </body>
              
              <body name="ant2_front_right_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="ant2_aux_2" size="0.08" type="capsule" rgba="0.8 0.2 0.2 1"/>
                <body name="ant2_aux_2" pos="-0.2 0.2 0">
                  <joint axis="0 0 1" name="ant2_hip_2" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                  <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="ant2_right_leg" size="0.08" type="capsule" rgba="0.8 0.2 0.2 1"/>
                  <body pos="-0.4 0.4 0">
                    <joint axis="1 1 0" name="ant2_ankle_2" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="ant2_right_ankle" size="0.08" type="capsule" rgba="0.8 0.2 0.2 1"/>
                  </body>
                </body>
              </body>
              
              <!-- 後脚 -->
              <body name="ant2_back_left_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="ant2_aux_3" size="0.08" type="capsule" rgba="0.8 0.2 0.2 1"/>
                <body name="ant2_aux_3" pos="0.2 -0.2 0">
                  <joint axis="0 0 1" name="ant2_hip_3" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                  <geom fromto="0.0 0.0 0.0 0.4 -0.4 0.0" name="ant2_back_leg" size="0.08" type="capsule" rgba="0.8 0.2 0.2 1"/>
                  <body pos="0.4 -0.4 0">
                    <joint axis="-1 -1 0" name="ant2_ankle_3" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 0.4 -0.4 0.0" name="ant2_back_left_ankle" size="0.08" type="capsule" rgba="0.8 0.2 0.2 1"/>
                  </body>
                </body>
              </body>
              
              <body name="ant2_back_right_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="ant2_aux_4" size="0.08" type="capsule" rgba="0.8 0.2 0.2 1"/>
                <body name="ant2_aux_4" pos="-0.2 -0.2 0">
                  <joint axis="0 0 1" name="ant2_hip_4" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                  <geom fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" name="ant2_right_back_leg" size="0.08" type="capsule" rgba="0.8 0.2 0.2 1"/>
                  <body pos="-0.4 -0.4 0">
                    <joint axis="1 -1 0" name="ant2_ankle_4" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" name="ant2_back_right_ankle" size="0.08" type="capsule" rgba="0.8 0.2 0.2 1"/>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
          
          <actuator>
            <!-- Agent 1 actuators -->
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant1_hip_1" gear="150"/>
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant1_ankle_1" gear="150"/>
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant1_hip_2" gear="150"/>
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant1_ankle_2" gear="150"/>
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant1_hip_3" gear="150"/>
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant1_ankle_3" gear="150"/>
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant1_hip_4" gear="150"/>
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant1_ankle_4" gear="150"/>
            
            <!-- Agent 2 actuators -->
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant2_hip_1" gear="150"/>
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant2_ankle_1" gear="150"/>
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant2_hip_2" gear="150"/>
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant2_ankle_2" gear="150"/>
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant2_hip_3" gear="150"/>
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant2_ankle_3" gear="150"/>
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant2_hip_4" gear="150"/>
            <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ant2_ankle_4" gear="150"/>
          </actuator>
        </mujoco>
        """
        
        self.xml_string = xml_string

    def _init_agent_geom_ids(self):
      """エージェントごとのgeom IDセットを初期化"""
      self.ant1_geom_ids = set()
      self.ant2_geom_ids = set()

      for geom_id in range(self.model.ngeom):
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not name:
          continue
        if name.startswith("ant1_"):
          self.ant1_geom_ids.add(geom_id)
        elif name.startswith("ant2_"):
          self.ant2_geom_ids.add(geom_id)

    def _init_body_ids(self):
        """トルソボディのIDを初期化"""
        self.ant1_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ant1_torso")
        self.ant2_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ant2_torso")

    def _is_body_flipped(self, body_id: int) -> bool:
        """胴体が上下反転しているかを判定

        data.xmat は各ボディの回転行列 (3x3, row-major) が並んだ配列。
        第3列がボディ座標系のZ軸（上方向）をワールド座標で表したものなので、
        そのZ成分が負なら上下さかさまとみなす。
        """
        if body_id is None or body_id < 0:
            return False
        # data.xmat は (nbody, 9) の2次元配列なので、body_id 行をそのまま取得
        try:
          R = self.data.xmat[body_id]
        except IndexError:
          return False

        # 行列要素 R[0..8]（row-major）のうち、第3列のZ成分が上方向
        up_z = R[8]
        return up_z < 0.0
        
    def reset(self, seed=None, options=None):
        """環境をリセット"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # MuJoCoモデルを読み込み
        self.model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.data = mujoco.MjData(self.model)

        # エージェントのgeom IDセットを初期化
        self._init_agent_geom_ids()

        # トルソボディIDを初期化
        self._init_body_ids()
        
        # ランダムな初期位置を設定（円の中）
        angle1 = np.random.uniform(0, 2 * np.pi)
        radius1 = np.random.uniform(0, self.arena_radius * 0.6)
        self.data.qpos[0] = radius1 * np.cos(angle1)  # ant1 x
        self.data.qpos[1] = radius1 * np.sin(angle1)  # ant1 y
        self.data.qpos[2] = 0.75  # ant1 z（地面から少し上）

        # Ant1の姿勢（free jointのクォータニオン）を立ち姿に初期化
        # qpos[3:7] = [w, x, y, z] として単位クォータニオンを設定
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        
        angle2 = angle1 + np.pi + np.random.uniform(-np.pi/4, np.pi/4)
        radius2 = np.random.uniform(0, self.arena_radius * 0.6)
        ant2_qpos_idx = 15  # ant2のqpos開始インデックス（3 + 4 + 8 = 15）
        self.data.qpos[ant2_qpos_idx] = radius2 * np.cos(angle2)  # ant2 x
        self.data.qpos[ant2_qpos_idx + 1] = radius2 * np.sin(angle2)  # ant2 y
        self.data.qpos[ant2_qpos_idx + 2] = 0.75  # ant2 z（地面から少し上）

        # Ant2の姿勢も同様に単位クォータニオンで初期化
        self.data.qpos[ant2_qpos_idx + 3 : ant2_qpos_idx + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        
        # 関節角度を初期化
        for i in range(8):
            self.data.qpos[7 + i] = 0.0  # ant1 joints
            self.data.qpos[22 + i] = 0.0  # ant2 joints
        
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        
        # Agent 1の観測を返す
        obs = self._get_obs(agent_id=1)
        info = {}
        
        return obs, info
    
    def step(self, action):
        """
        1ステップ実行
        action: dict形式 {"agent1": action1, "agent2": action2}
        それぞれのactionは8次元のnumpy配列
        """
        # Agent 1とAgent 2の行動を適用
        if isinstance(action, dict):
            action1 = np.clip(action.get("agent1", np.zeros(8)), -1, 1)
            action2 = np.clip(action.get("agent2", np.zeros(8)), -1, 1)
        else:
            # 単一エージェント用（agent1のみ）
            action1 = np.clip(action, -1, 1)
            action2 = np.zeros(8)
        
        # アクチュエータに行動を設定
        self.data.ctrl[:8] = action1
        self.data.ctrl[8:16] = action2
        
        # シミュレーション実行
        mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        
        # 報酬と終了判定
        reward1, reward2, terminated, info = self._compute_reward_and_done()
        
        truncated = self.current_step >= self.max_episode_steps
        
        # Agent 1の観測を返す
        obs = self._get_obs(agent_id=1)
        
        # 両エージェントの報酬を含める
        info["agent1_reward"] = reward1
        info["agent2_reward"] = reward2
        
        return obs, reward1, terminated, truncated, info
    
    def _get_obs(self, agent_id=1):
      """観測を取得"""
      if agent_id == 1:
        # Agent 1の観測
        # 自分の位置 (x, y)
        my_pos = self.data.qpos[0:2].copy()
        # 自分の関節角度 (8次元)
        my_joints = self.data.qpos[7:15].copy()
        # 自分の関節角速度 (8次元)
        # qvel: ant1_root(6) + ant1_joints(8) + ant2_root(6) + ant2_joints(8)
        my_joint_vel = self.data.qvel[6:14].copy()
        # 自分の関節トルク (8次元)
        my_actuator_forces = self.data.actuator_force[:8].copy()
        # 相手の位置 (x, y)
        opp_pos = self.data.qpos[15:17].copy()
        # 相手の関節角度 (8次元)
        opp_joints = self.data.qpos[22:30].copy()
        # 相手の関節角速度 (8次元)
        opp_joint_vel = self.data.qvel[20:28].copy()

        obs = np.concatenate([
          my_pos,
          my_joints,
          my_joint_vel,
          my_actuator_forces,
          opp_pos,
          opp_joints,
          opp_joint_vel,
        ])
      else:
        # Agent 2の観測
        my_pos = self.data.qpos[15:17].copy()
        my_joints = self.data.qpos[22:30].copy()
        # ant2の関節角速度は qvel[20:28]
        my_joint_vel = self.data.qvel[20:28].copy()
        my_actuator_forces = self.data.actuator_force[8:16].copy()
        opp_pos = self.data.qpos[0:2].copy()
        opp_joints = self.data.qpos[7:15].copy()
        # 相手（ant1）の関節角速度は qvel[6:14]
        opp_joint_vel = self.data.qvel[6:14].copy()

        obs = np.concatenate([
          my_pos,
          my_joints,
          my_joint_vel,
          my_actuator_forces,
          opp_pos,
          opp_joints,
          opp_joint_vel,
        ])

      return obs.astype(np.float32)
    
    def _compute_reward_and_done(self):
        """報酬と終了判定を計算"""
        # Agent 1とAgent 2の位置を取得
        ant1_pos = self.data.qpos[0:2]
        ant2_pos = self.data.qpos[15:17]
        
        # 円からの距離を計算
        ant1_dist = np.linalg.norm(ant1_pos)
        ant2_dist = np.linalg.norm(ant2_pos)
        
        # ひっくり返ったかをチェック（胴体の上下が反転しているか）
        ant1_flipped = self._is_body_flipped(self.ant1_body_id)
        ant2_flipped = self._is_body_flipped(self.ant2_body_id)
        
        # 円の外に出たかをチェック
        ant1_out = ant1_dist > self.arena_radius
        ant2_out = ant2_dist > self.arena_radius

        # シミュレーション安定化のため、開始直後の数ステップは終了判定を無効化
        # （初期姿勢の微小な揺れや数値誤差で即終了しないようにする）
        if self.current_step <= 5:
          ant1_flipped = False
          ant2_flipped = False
          ant1_out = False
          ant2_out = False

        # 時間経過のペナルティのみ
        reward1 = self.time_penalty
        reward2 = self.time_penalty
        terminated = False
        info = {}
        
        # 勝敗判定（勝者に +1000, 敗者に -1000）
        if ant1_flipped or ant1_out:
            # Agent 1が負け
          reward1 = -1000.0
          reward2 = 1000.0
          terminated = True
          info["winner"] = "agent2"
        elif ant2_flipped or ant2_out:
            # Agent 2が負け
          reward1 = 1000.0
          reward2 = -1000.0
          terminated = True
          info["winner"] = "agent1"
        
        return reward1, reward2, terminated, info
    
    def render(self):
        """レンダリング（オプション）"""
        if self.render_mode == "human":
            # 人間用のレンダリング（実装は省略）
            pass
        elif self.render_mode == "rgb_array":
            # RGB配列でのレンダリング（実装は省略）
            pass
        return None
    
    def close(self):
        """環境を閉じる"""
        pass
