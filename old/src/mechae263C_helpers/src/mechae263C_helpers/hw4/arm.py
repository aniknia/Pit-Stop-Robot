import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydrake.systems.framework import (
    Diagram,
    DiagramBuilder,
    LeafSystem,
    BasicVector,
    Context,
)
from pydrake.systems.primitives import Integrator, MatrixGain

from mechae263C_helpers.drake import LinearCombination


@dataclass(frozen=True)
class Link:
    length: float
    mass: float
    joint_axis_to_com_dist: float
    com_inertia: float


@dataclass(frozen=True)
class Motor:
    mass: float
    com_inertia: float
    transmission_ratio: float


@dataclass(frozen=True)
class AugmentedLink:
    mass: float
    first_inertia_moment: float
    link_frame_inertia: float


class CoriolisCentripetal(LeafSystem):
    def __init__(self, k3: float):
        super().__init__()
        self.k3 = float(k3)

        self.DeclareVectorInputPort(name="q", size=2)
        self.DeclareVectorInputPort(name="dq", size=2)
        self.DeclareVectorOutputPort(name="Cdq", size=2, calc=self.calc_output)

    def calc_output(self, context: Context, output: BasicVector):
        q = self.get_input_port(0).Eval(context)
        dq = self.get_input_port(1).Eval(context)
        q0, q1 = q[0], q[1]
        dq0, dq1 = dq[0], dq[1]

        Cdq = np.zeros((2,), dtype=np.double)

        x0 = self.k3 * math.sin(q1)
        x1 = x0 * dq1
        x2 = 2 * x1 * dq0
        x3 = x1 * dq1
        Cdq[0] = x2 + x3
        Cdq[1] = -x0 * dq0 * dq0

        output.set_value(Cdq)


class ResolvedAcceleration(LeafSystem):
    def __init__(
        self,
        link_lens: Sequence[float],
        dyn_params: NDArray[np.double],
        transmission_ratio1: float,
    ):
        super().__init__()

        self.DeclareVectorInputPort(name="net_torque", size=2)
        self.DeclareVectorInputPort(name="q", size=2)
        self.DeclareVectorOutputPort(name="invBq", size=2, calc=self.calc_output)

        self.link_lens = np.asarray(link_lens)
        self.dyn_params = np.asarray(dyn_params)
        self.transmission_ratio1 = transmission_ratio1

    def calc_output(self, context: Context, output: BasicVector):
        tau = self.get_input_port(0).Eval(context)
        q = self.get_input_port(1).Eval(context)
        _, q1 = q[0], q[1]

        ddq = np.linalg.inv(self.calc_inertia_mat(math.cos(q1))) @ tau

        output.set_value(ddq)

    def calc_inertia_mat(self, cos_q1: float):
        inertia_mat = np.zeros((2, 2), dtype=np.double)

        inertia_mat[0, 0] = (
            self.link_lens[0] * self.dyn_params[0]
            + self.dyn_params[1]
            + (self.link_lens[1] + 2 * self.link_lens[0] * cos_q1) * self.dyn_params[2]
            + self.dyn_params[3]
        )

        inertia_mat[0, 1] = inertia_mat[1, 0] = (
            (self.link_lens[1] + self.link_lens[0] * cos_q1) * self.dyn_params[2]
            + self.dyn_params[3]
            + self.transmission_ratio1 * self.dyn_params[4]
        )

        inertia_mat[1, 1] = (
            self.link_lens[1] * self.dyn_params[2]
            + self.dyn_params[3]
            + self.transmission_ratio1 * self.transmission_ratio1 * self.dyn_params[4]
        )

        return inertia_mat


class Gravity(LeafSystem):
    def __init__(self, dyn_params: NDArray[np.double], g: float = 9.81):
        super().__init__()

        self.DeclareVectorInputPort(name="q", size=2)
        self.DeclareVectorOutputPort(name="g", size=2, calc=self.calc_output)

        self.k1, self.k2 = g * dyn_params[0], g * dyn_params[2]

    def calc_output(self, context: Context, output: BasicVector):
        q = self.get_input_port().Eval(context)
        q0, q1 = q[0], q[1]

        k1_cos_q0 = self.k1 * math.cos(q0)
        k2_cos_q01 = self.k2 * math.cos(q0 + q1)

        gravity = np.zeros((2,), dtype=np.double)
        gravity[0] = k1_cos_q0 + k2_cos_q01
        gravity[1] = k2_cos_q01

        output.set_value(gravity)


class Arm(Diagram):
    def __init__(self):
        super().__init__()

        K_r = 50 * np.eye(2)
        F_v = K_r @ np.diag([0.01, 0.01]) @ K_r
        self.F_v = np.asarray(F_v, dtype=np.double)

        m = 9
        L = 1
        self.links = [
            Link(
                length=L,
                mass=m,
                joint_axis_to_com_dist=L / 2,
                com_inertia=m * L ** 2 / 3,
            ),
            Link(
                length=L,
                mass=m,
                joint_axis_to_com_dist=L / 2,
                com_inertia=m * L ** 2 / 3,
            ),
        ]
        self.motors = [
            Motor(mass=1.0, com_inertia=0.007, transmission_ratio=K_r[0, 0]),
            Motor(mass=1.0, com_inertia=0.007, transmission_ratio=K_r[1, 1]),
        ]

        m1, m2 = self.links[0].mass, self.links[1].mass
        a1, a2 = self.links[0].length, self.links[1].length
        l1, l2 = (
            self.links[0].joint_axis_to_com_dist,
            self.links[1].joint_axis_to_com_dist,
        )
        I1, I2 = self.links[0].com_inertia, self.links[1].com_inertia

        self.augmented_links = [
            AugmentedLink(
                mass=m1 + self.motors[1].mass,
                first_inertia_moment=m1 * (l1 - a1),
                link_frame_inertia=I1
                + m1 * (l1 - a1) ** 2
                + self.motors[0].com_inertia,
            ),
            AugmentedLink(
                mass=m2,
                first_inertia_moment=m2 * (l2 - a2),
                link_frame_inertia=I2 + m2 * (l2 - a2) ** 2,
            ),
        ]

        self.dyn_params = np.asarray(
            [
                a1 * self.augmented_links[0].mass
                + self.augmented_links[0].first_inertia_moment
                + a1 * self.augmented_links[1].mass,
                a1 * self.augmented_links[0].first_inertia_moment
                + self.augmented_links[0].link_frame_inertia
                + self.motors[0].transmission_ratio ** 2 * self.motors[0].com_inertia,
                a2 * self.augmented_links[1].mass
                + self.augmented_links[1].first_inertia_moment,
                a2 * self.augmented_links[1].first_inertia_moment
                + self.augmented_links[1].link_frame_inertia,
                self.motors[1].com_inertia,
            ]
        )

        diagram_builder: DiagramBuilder = DiagramBuilder()

        net_torque: LinearCombination = diagram_builder.AddNamedSystem(
            "net_torque",
            LinearCombination(input_coeffs=(1, -1, -1, -1), input_shapes=(2,)),
        )
        integrator0: LinearCombination = diagram_builder.AddNamedSystem(
            "integrator0", Integrator(2)
        )
        integrator1: LinearCombination = diagram_builder.AddNamedSystem(
            "integrator1", Integrator(2)
        )
        coriolis_centrifugal = diagram_builder.AddNamedSystem(
            "coriolis_centrifugal",
            CoriolisCentripetal(k3=-self.links[0].length * self.dyn_params[2]),
        )
        motor_friction = diagram_builder.AddNamedSystem(
            "motor_friction", MatrixGain(self.F_v)
        )

        self.resolved_accel = diagram_builder.AddNamedSystem(
            "resolved_accel",
            ResolvedAcceleration(
                link_lens=[self.links[0].length, self.links[1].length],
                dyn_params=self.dyn_params,
                transmission_ratio1=self.motors[1].transmission_ratio,
            ),
        )

        gravity = diagram_builder.AddNamedSystem(
            "gravity", Gravity(dyn_params=self.dyn_params)
        )

        diagram_builder.Connect(
            net_torque.get_output_port(), self.resolved_accel.get_input_port(0)
        )
        diagram_builder.Connect(
            self.resolved_accel.get_output_port(), integrator0.get_input_port()
        )
        diagram_builder.Connect(
            integrator0.get_output_port(), motor_friction.get_input_port()
        )
        diagram_builder.Connect(
            integrator0.get_output_port(), integrator1.get_input_port()
        )
        diagram_builder.Connect(
            integrator0.get_output_port(), coriolis_centrifugal.get_input_port(1)
        )
        diagram_builder.Connect(
            integrator1.get_output_port(), coriolis_centrifugal.get_input_port(0)
        )

        diagram_builder.Connect(
            coriolis_centrifugal.get_output_port(), net_torque.get_input_port(1)
        )
        diagram_builder.Connect(
            motor_friction.get_output_port(), net_torque.get_input_port(2)
        )
        diagram_builder.Connect(integrator1.get_output_port(), gravity.get_input_port())
        diagram_builder.Connect(
            integrator0.get_output_port(), self.resolved_accel.get_input_port(1)
        )
        diagram_builder.Connect(gravity.get_output_port(), net_torque.get_input_port(3))

        diagram_builder.ExportInput(net_torque.get_input_port(0), name="tau")
        diagram_builder.ExportOutput(integrator0.get_output_port(), name="dq")
        diagram_builder.ExportOutput(integrator1.get_output_port(), name="q")

        diagram_builder.BuildInto(self)
