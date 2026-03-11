from ._base_task import Base_Task
from .utils import *
import sapien
import math


class replay_onlyrobot(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        pass

    def take_action(self, action, action_type='qpos'):  # action_type: qpos or ee
        self.take_action_cnt += 1
        print(f"step: \033[92m{self.take_action_cnt}\033[0m", end="\r")

        self._update_render()
        if self.render_freq:
            self.viewer.render()

        actions = np.array([action])  # [1, 14]
        left_jointstate = self.robot.get_left_arm_jointState()
        right_jointstate = self.robot.get_right_arm_jointState()
        left_arm_dim = len(left_jointstate) - 1
        right_arm_dim = len(right_jointstate) - 1
        current_jointstate = np.array(left_jointstate + right_jointstate)

        left_arm_actions, left_gripper_actions, left_current_qpos, left_path = ([],[],[],[])
        right_arm_actions, right_gripper_actions, right_current_qpos, right_path = ([],[],[],[])
        # gripper部分需要归一化。
        # 采集数据中qpos：夹爪0.0007闭合，0.069张开;有时候也是0.00063,到0.06888
        # robotwin中夹爪：0闭合，1张开
        actions[:, left_arm_dim] = (actions[:, left_arm_dim] - 0.0005) / (0.069 - 0.0005)
        actions[:, left_arm_dim] = np.clip(actions[:, left_arm_dim], 0.0, 1.0)
        actions[:, left_arm_dim + right_arm_dim + 1] = (actions[:, left_arm_dim + right_arm_dim + 1] - 0.0005) / (0.069 - 0.0005)
        actions[:, left_arm_dim + right_arm_dim + 1] = np.clip(actions[:, left_arm_dim + right_arm_dim + 1], 0.0, 1.0)
        
        left_arm_actions, left_gripper_actions = (
            actions[:, :left_arm_dim],  # [1, 6]
            actions[:, left_arm_dim],  # [1, 1]
        )
        right_arm_actions, right_gripper_actions = (
            actions[:, left_arm_dim + 1:left_arm_dim + right_arm_dim + 1],
            actions[:, left_arm_dim + right_arm_dim + 1],
        )
        left_current_qpos, right_current_qpos = (
            current_jointstate[:left_arm_dim],
            current_jointstate[left_arm_dim + 1:left_arm_dim + right_arm_dim + 1],
        )
        left_current_gripper, right_current_gripper = (
            current_jointstate[left_arm_dim:left_arm_dim + 1],
            current_jointstate[left_arm_dim + right_arm_dim + 1:left_arm_dim + right_arm_dim + 2],
        )

        left_path = np.vstack((left_current_qpos, left_arm_actions))
        left_gripper_path = np.hstack((left_current_gripper, left_gripper_actions))
        right_path = np.vstack((right_current_qpos, right_arm_actions))
        right_gripper_path = np.hstack((right_current_gripper, right_gripper_actions))

        # ========== TOPP ==========
        topp_left_flag, topp_right_flag = True, True

        times, left_pos, left_vel, acc, duration = (self.robot.left_mplib_planner.TOPP(left_path, 1 / 250, verbose=True))
        left_result = dict()
        left_result["position"], left_result["velocity"] = left_pos, left_vel
        left_n_step = left_result["position"].shape[0]

        if left_n_step == 0:
            topp_left_flag = False
            left_n_step = 50  # fixed

        times, right_pos, right_vel, acc, duration = (self.robot.right_mplib_planner.TOPP(right_path, 1 / 250, verbose=True))
        right_result = dict()
        right_result["position"], right_result["velocity"] = right_pos, right_vel
        right_n_step = right_result["position"].shape[0]

        if right_n_step == 0:
            topp_right_flag = False
            right_n_step = 50  # fixed

        # ========== Gripper ==========

        left_mod_num = left_n_step % len(left_gripper_actions)
        right_mod_num = right_n_step % len(right_gripper_actions)
        left_gripper_step = [0] + [
            left_n_step // len(left_gripper_actions) + (1 if i < left_mod_num else 0)
            for i in range(len(left_gripper_actions))
        ]
        right_gripper_step = [0] + [
            right_n_step // len(right_gripper_actions) + (1 if i < right_mod_num else 0)
            for i in range(len(right_gripper_actions))
        ]

        left_gripper = []
        for gripper_step in range(1, left_gripper_path.shape[0]):
            region_left_gripper = np.linspace(
                left_gripper_path[gripper_step - 1],
                left_gripper_path[gripper_step],
                left_gripper_step[gripper_step] + 1,
            )[1:]
            left_gripper = left_gripper + region_left_gripper.tolist()
        left_gripper = np.array(left_gripper)

        right_gripper = []
        for gripper_step in range(1, right_gripper_path.shape[0]):
            region_right_gripper = np.linspace(
                right_gripper_path[gripper_step - 1],
                right_gripper_path[gripper_step],
                right_gripper_step[gripper_step] + 1,
            )[1:]
            right_gripper = right_gripper + region_right_gripper.tolist()
        right_gripper = np.array(right_gripper)

        now_left_id, now_right_id = 0, 0

        # ========== Control Loop ==========
        while now_left_id < left_n_step or now_right_id < right_n_step:

            if (now_left_id < left_n_step and now_left_id / left_n_step <= now_right_id / right_n_step):
                if topp_left_flag:
                    self.robot.set_arm_joints(
                        left_result["position"][now_left_id],
                        left_result["velocity"][now_left_id],
                        "left",
                    )
                self.robot.set_gripper(left_gripper[now_left_id], "left")

                now_left_id += 1

            if (now_right_id < right_n_step and now_right_id / right_n_step <= now_left_id / left_n_step):
                if topp_right_flag:
                    self.robot.set_arm_joints(
                        right_result["position"][now_right_id],
                        right_result["velocity"][now_right_id],
                        "right",
                    )
                self.robot.set_gripper(right_gripper[now_right_id], "right")

                now_right_id += 1

            self.scene.step()
            self._update_render()

        self._update_render()
        if self.render_freq:  # UI
            self.viewer.render()
            
            
    def take_action_simple(self, action, action_type='qpos'):
        # 不进行规划和任何插值，使用self.robot.set_qpos(full_qpos)直接设置到位
        self.take_action_cnt += 1
        print(f"step: \033[92m{self.take_action_cnt}\033[0m", end="\r")
        actions = np.array([action])  # [1, 14]
        left_jointstate = self.robot.get_left_arm_jointState()
        right_jointstate = self.robot.get_right_arm_jointState()
        left_arm_dim = len(left_jointstate) - 1
        right_arm_dim = len(right_jointstate) - 1
        current_jointstate = np.array(left_jointstate + right_jointstate)
        # gripper部分需要归一化
        # 采集数据中qpos：夹爪0.0007闭合，0.069张开;有时候也是0.00063,到0.06888
        # robotwin中夹爪：0闭合，1张开
        # actions[:, left_arm_dim] = (actions[:, left_arm_dim] - 0.0005) / (0.069 - 0.0005)
        # actions[:, left_arm_dim] = np.clip(actions[:, left_arm_dim], 0.0, 1.0)
        # actions[:, left_arm_dim + right_arm_dim + 1] = (actions[:, left_arm_dim + right_arm_dim + 1] - 0.0005) / (
        #     0.069 - 0.0005
        # )
        actions[:, left_arm_dim + right_arm_dim + 1] = np.clip(
            actions[:, left_arm_dim + right_arm_dim + 1], 0.0, 1.0
        )

        left_arm_actions = actions[0, :left_arm_dim]
        left_gripper_action = actions[0, left_arm_dim]
        right_arm_actions = actions[0, left_arm_dim + 1:left_arm_dim + right_arm_dim + 1]
        right_gripper_action = actions[0, left_arm_dim + right_arm_dim + 1]

        def _apply_qpos_to_entity(entity, arm_joints, gripper_joints, arm_actions, gripper_action, gripper_scale):
            qpos = entity.get_qpos().copy()
            active_joints = entity.get_active_joints()

            for idx, joint in enumerate(arm_joints):
                qpos[active_joints.index(joint)] = arm_actions[idx]

            if gripper_joints:
                real_gripper_val = gripper_scale[0] + gripper_action * (gripper_scale[1] - gripper_scale[0])
                for joint, scale, bias in gripper_joints:
                    qpos[active_joints.index(joint)] = real_gripper_val * scale + bias

            entity.set_qpos(qpos)

        if getattr(self.robot, "is_dual_arm", False):
            entity = self.robot.left_entity
            qpos = entity.get_qpos().copy()
            active_joints = entity.get_active_joints()

            for idx, joint in enumerate(self.robot.left_arm_joints):
                qpos[active_joints.index(joint)] = left_arm_actions[idx]
            for idx, joint in enumerate(self.robot.right_arm_joints):
                qpos[active_joints.index(joint)] = right_arm_actions[idx]

            if self.robot.left_gripper:
                left_real_gripper_val = self.robot.left_gripper_scale[0] + left_gripper_action * (
                    self.robot.left_gripper_scale[1] - self.robot.left_gripper_scale[0]
                )
                for joint, scale, bias in self.robot.left_gripper:
                    qpos[active_joints.index(joint)] = left_real_gripper_val * scale + bias

            if self.robot.right_gripper:
                right_real_gripper_val = self.robot.right_gripper_scale[0] + right_gripper_action * (
                    self.robot.right_gripper_scale[1] - self.robot.right_gripper_scale[0]
                )
                for joint, scale, bias in self.robot.right_gripper:
                    qpos[active_joints.index(joint)] = right_real_gripper_val * scale + bias

            entity.set_qpos(qpos)
        else:
            _apply_qpos_to_entity(
                self.robot.left_entity,
                self.robot.left_arm_joints,
                self.robot.left_gripper,
                left_arm_actions,
                left_gripper_action,
                self.robot.left_gripper_scale,
            )
            _apply_qpos_to_entity(
                self.robot.right_entity,
                self.robot.right_arm_joints,
                self.robot.right_gripper,
                right_arm_actions,
                right_gripper_action,
                self.robot.right_gripper_scale,
            )

        self.robot.left_gripper_val = float(left_gripper_action)
        self.robot.right_gripper_val = float(right_gripper_action)

        self.scene.step()
        self._update_render()
        if self.render_freq:
            self.viewer.render()

