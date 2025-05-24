from functools import partial
from itertools import zip_longest
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from pydrake.common.value import Value, AbstractValue
from pydrake.systems.framework import LeafSystem, Context, BasicVector

ShapeTypeOrNone = tuple[int, ...] | None


class LinearCombination(LeafSystem):
    input_coeffs: tuple[int | float | NDArray[np.double], ...]
    input_shape: tuple[int, ...] | None

    def __init__(
        self,
        input_coeffs: tuple[
            int | float | NDArray[np.double] | Sequence[int | float], ...
        ],
        input_shapes: tuple[ShapeTypeOrNone, ...] | ShapeTypeOrNone = None,
    ):
        super().__init__()

        if len(input_coeffs) == 0:
            raise ValueError(
                "`LinearCombination.__init__` expects at least one input coefficient "
                "but received 0"
            )

        if input_shapes is not None and len(input_shapes) == 0:
            raise ValueError(
                "`LinearCombination.__init__` expects at least one input shape "
                "when argument `input_shapes` is not `None` but received 0"
            )

        self.input_coeffs = input_coeffs

        self.input_shapes = input_shapes
        self.output_shape = (
            (1,)
            if input_shapes is None
            else (input_coeffs[0] * np.empty(input_shapes[0])).shape
        )

        if input_shapes is None:
            for i in range(len(input_coeffs)):
                self.DeclareVectorInputPort(name=f"u{i}", size=1)
        elif isinstance(input_shapes, tuple) and isinstance(input_shapes[0], tuple):
            if len(input_coeffs) >= len(input_shapes):
                zip_fn = partial(zip_longest, fillvalue=None)
            else:
                zip_fn = zip

            for i, (_, input_shape) in enumerate(zip_fn(input_coeffs, input_shapes)):
                model_value = 0.0 if input_shape is None else np.empty(input_shape)

                self.DeclareAbstractInputPort(
                    name=f"u{i}", model_value=AbstractValue.Make(model_value)
                )
        else:
            input_shape = input_shapes[0]
            for i in range(len(input_coeffs)):
                self.DeclareVectorInputPort(name=f"u{i}", size=input_shape)

        # TODO: Add AbstractOutputPort if output should be matrix
        output_size = self.output_shape[0]

        self.DeclareVectorOutputPort("out", size=output_size, calc=self.calc_output)

    def calc_output(self, context: Context, output: BasicVector):
        linear_combination = np.zeros(self.output_shape)

        for i, input_coeff in enumerate(self.input_coeffs):
            linear_combination += input_coeff * np.asarray(
                self.get_input_port(i).Eval(context)
            )

        output.set_value(linear_combination)
