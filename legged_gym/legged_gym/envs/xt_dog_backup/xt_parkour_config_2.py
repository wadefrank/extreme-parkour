# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# python legged_gym/legged_gym/scripts/train.py \
# --task xt_dog_stage2 \
# --exptid your_stage2_run \
# --resume \
# --load_run your_stage1_run

from .xt_parkour_config import XTDogParkourCfg, XTDogParkourCfgPPO


class XTDogParkourStage2Cfg(XTDogParkourCfg):
    class terrain(XTDogParkourCfg.terrain):
        # 第二阶段：在保留跳跃地形的前提下，恢复更多平地和横向随机，逼策略学习泛化。
        y_range = [-0.3, 0.3]
        height = [0.02, 0.05]
        max_init_terrain_level = 4
        num_rows = 10

        terrain_dict = {"smooth slope": 0.,
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "rough stairs up": 0.,
                        "rough stairs down": 0.,
                        "discrete": 0.,
                        "stepping stones": 0.0,
                        "gaps": 0.,
                        "smooth flat": 0.,
                        "pit": 0.0,
                        "wall": 0.0,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.15,
                        "parkour_hurdle": 0.25,
                        "parkour_flat": 0.20,
                        "parkour_step": 0.20,
                        "parkour_gap": 0.20,
                        "demo": 0.0,}
        terrain_proportions = list(terrain_dict.values())

        # 第二阶段把障碍的横向偏移和落脚带变化再放开一些。
        parkour_hurdle_half_valid_width = [0.50, 0.95]
        parkour_flat_half_valid_width = [0.55, 1.05]
        parkour_step_half_valid_width = [0.55, 1.00]
        parkour_gap_half_valid_width = [0.65, 1.20]

    class domain_rand(XTDogParkourCfg.domain_rand):
        # 第二阶段逐步把随机化加回来，避免只会解一种模板地形。
        added_mass_range = [0., 4.]
        added_com_range = [-0.1, 0.1]
        motor_strength_range = [0.85, 1.15]
        push_robots = True


class XTDogParkourStage2CfgPPO(XTDogParkourCfgPPO):
    class runner(XTDogParkourCfgPPO.runner):
        experiment_name = 'parkour_xt_dog_stage2'
