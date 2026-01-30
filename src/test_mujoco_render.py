"""MuJoCoのレンダリング確認用・最小スクリプト

Dockerコンテナ内で以下のように実行してください:

    cd /workspace/src
            python test_mujoco_render.py
            
            ウィンドウが立ち上がり、2体のAntが適当に動けばOKです。"""

import os
import sys
import time

# プロジェクトルートをパスに追加\nsys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco  # type: ignore
import mujoco.viewer  # type: ignore
from sumo_env import AntSumoEnv

def main():
    # 環境作成
    env = AntSumoEnv(arena_radius=5.0, max_episode_steps=1000)
    obs, _ = env.reset()

    print("[INFO] AntSumoEnv initialized")
    print(f"[INFO] Observation dim: {env.observation_space.shape[0]}")
    print(f"[INFO] Action dim: {env.action_space.shape[0]}")

    # MuJoCoビューワ起動
    print("[INFO] Launching MuJoCo viewer...")
    viewer = mujoco.viewer.launch_passive(env.model, env.data)

    # カメラの簡単な設定
    viewer.cam.distance = 15.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90

    # 数百ステップだけランダムに動かす
    fps = 60
    num_steps = 600
    for step in range(num_steps):
        # ランダム行動で2体を動かす
        action1 = env.action_space.sample()*2
        action2 = env.action_space.sample()
        obs, reward1, terminated, truncated, info = env.step({"agent1": action1, "agent2": action2})

        # 両エージェントの報酬を表示
        r1 = reward1
        r2 = info.get("agent2_reward", 0.0)
        print(f"[STEP {step:04d}] agent1_reward={r1:+.3f}, agent2_reward={r2:+.3f}")

        # レンダリング
        viewer.sync()
        time.sleep(1.0 / fps)

        if terminated or truncated:
            print("[INFO] episode finished (terminated=%s, truncated=%s, winner=%s)" % (
                terminated,
                truncated,
                info.get("winner", "none"),
            ))
            # 1エピソードだけ見られれば十分なので抜ける
            break

    print("[INFO] Closing viewer and environment...")
    viewer.close()
    env.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()