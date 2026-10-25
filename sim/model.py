"""実機（3Dプリント済み）のシャクトリムシの MuJoCo モデル。

  [前の節]──J1──[中の節]──J2──[後の節]        進行方向 = +x（前の節の向き）
   70mm         124mm        70mm

寸法・質量はすべて実機の実測値。関節は両方 pitch で、
**プラス＝先端が持ち上がる**向きに揃えてある（実機は Dynamixel の Drive Mode 反転で合わせる）。

中の節は底面がべったり接地し、前後の節は 0° のとき床から 6mm 浮く。
両方を -90° にすると中の節が 51mm 浮く（＝脚で体を持ち上げられる）。
"""

from dataclasses import dataclass

import mujoco


@dataclass
class Params:
  # --- 実測（mm を m に直したもの）---
  mid_len: float = 0.124
  mid_w: float = 0.058
  mid_h: float = 0.036
  mid_mass: float = 0.1796  # M5Stack Core2 + バッテリー + 基板 + シェル
  end_len: float = 0.070
  end_w: float = 0.074
  end_h: float = 0.024
  end_mass: float = 0.0468  # XL330 1 個 + シェル
  axis_z: float = 0.019  # 床から関節軸までの高さ
  # --- 同定対象（実機のログで詰める）---
  friction: float = 0.3  # PLA と木の床。滑りやすいとのこと
  frictionloss: float = 0.03  # 「自重で落ちてこない」から下限 0.0161 N·m 以上
  damping: float = 0.002
  armature: float = 0.0005  # 288:1 の反射慣性。要同定
  # --- 数値 ---
  timestep: float = 0.001
  noslip: int = 10  # これが無いと frictionloss が効かずクリープする
  joint_range: float = 90.0


def _box_inertia(m: float, lx: float, ly: float, lz: float) -> str:
  ix = m * (ly**2 + lz**2) / 12
  iy = m * (lx**2 + lz**2) / 12
  iz = m * (lx**2 + ly**2) / 12
  return f"{ix:.6e} {iy:.6e} {iz:.6e}"


def _end(p: Params, name: str, s: int, material: str) -> str:
  """前後の節。s=+1 が前（+x 方向に伸びる）、s=-1 が後ろ。"""
  r = p.end_h / 2  # 先端の丸みの半径 = 節の高さの半分
  dz = p.mid_h / 2 - p.axis_z  # 軸から節の中心までの高さの差（-1mm）
  box_half = (p.end_len - r) / 2
  axis = "0 -1 0" if s > 0 else "0 1 0"  # どちらも「プラス＝持ち上げ」
  return f"""
      <body name="{name}" pos="{s * p.mid_len / 2} 0 {p.axis_z - p.mid_h / 2}">
        <joint name="{"J1" if s > 0 else "J2"}" type="hinge" axis="{axis}" range="-{p.joint_range} {p.joint_range}" damping="{p.damping}" armature="{p.armature}" frictionloss="{p.frictionloss}"/>
        <inertial pos="{s * p.end_len / 2} 0 {dz}" mass="{p.end_mass}" diaginertia="{_box_inertia(p.end_mass, p.end_len, p.end_w, p.end_h)}"/>
        <geom name="{name}_box" type="box" pos="{s * box_half} 0 {dz}" size="{box_half} {p.end_w / 2} {r}" material="{material}"/>
        <geom name="{name}_tip" type="cylinder" pos="{s * (p.end_len - r)} 0 {dz}" size="{r} {p.end_w / 2}" zaxis="0 1 0" material="{material}"/>
      </body>"""


def build_xml(p: Params) -> str:
  return f"""
<mujoco model="inchworm_v2">
  <option timestep="{p.timestep}" integrator="implicitfast" noslip_iterations="{p.noslip}"/>
  <visual>
    <global offwidth="1600" offheight="900"/>
    <headlight ambient="0.32 0.32 0.32" diffuse="0.35 0.35 0.35" specular="0.08 0.08 0.08"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.58 0.70 0.82" rgb2="0.93 0.95 0.96" width="512" height="3072"/>
    <texture name="grid" type="2d" builtin="checker" mark="edge" rgb1="0.62 0.66 0.65" rgb2="0.52 0.57 0.56" markrgb="0.40 0.44 0.43" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="12 12" texuniform="true" reflectance="0.05"/>
    <material name="m_mid" rgba="0.25 0.45 0.85 1" specular="0.3" shininess="0.4"/>
    <material name="m_front" rgba="0.90 0.50 0.18 1" specular="0.3" shininess="0.4"/>
    <material name="m_rear" rgba="0.25 0.65 0.38 1" specular="0.3" shininess="0.4"/>
  </asset>
  <default>
    <geom friction="{p.friction} 0.005 0.0001"/>
  </default>
  <worldbody>
    <light name="key" pos="0.4 -0.5 0.9" dir="-0.3 0.4 -1" directional="true" diffuse="0.45 0.45 0.45" castshadow="true"/>
    <light name="fill" pos="-0.5 0.5 0.7" dir="0.4 -0.4 -1" directional="true" diffuse="0.18 0.18 0.18" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="grid"/>
    <body name="mid" pos="0 0 {p.mid_h / 2}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{p.mid_mass}" diaginertia="{_box_inertia(p.mid_mass, p.mid_len, p.mid_w, p.mid_h)}"/>
      <geom name="mid_box" type="box" size="{p.mid_len / 2} {p.mid_w / 2} {p.mid_h / 2}" material="m_mid"/>
      <site name="imu" pos="0 0 {p.mid_h / 2}" size="0.004"/>{_end(p, "front", +1, "m_front")}{_end(p, "rear", -1, "m_rear")}
    </body>
  </worldbody>
  <actuator>
    <motor name="J1" joint="J1" ctrlrange="-0.52 0.52"/>
    <motor name="J2" joint="J2" ctrlrange="-0.52 0.52"/>
  </actuator>
  <sensor>
    <gyro name="gyro" site="imu"/>
    <jointpos name="q1" joint="J1"/>
    <jointpos name="q2" joint="J2"/>
  </sensor>
</mujoco>
"""


def build(p: Params | None = None):
  p = p or Params()
  m = mujoco.MjModel.from_xml_string(build_xml(p))
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  return m, d
