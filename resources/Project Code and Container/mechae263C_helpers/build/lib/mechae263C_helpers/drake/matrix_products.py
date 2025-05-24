import numpy as np
from pydrake.common.value import AbstractValue, Value
from pydrake.systems.framework import LeafSystem, Context, BasicVector, InputPort


class MatrixVectorProduct(LeafSystem):
    matrix_shape: tuple[int, int]

    def __init__(self, matrix_shape: tuple[int, int]):
        super().__init__()

        if len(matrix_shape) != 2:
            raise ValueError(
                "`MatrixVectorProduct.__init__` expects the a tuple of size 2 with the "
                f"shape of the matrix, but received {matrix_shape}"
            )

        self.matrix_shape = matrix_shape

        self.DeclareAbstractInputPort(
            name=f"K",
            model_value=AbstractValue.Make(
                np.empty(self.matrix_shape, dtype=np.double)
            ),
        )
        self.DeclareVectorInputPort(name=f"u", size=self.matrix_shape[1])
        self.DeclareVectorOutputPort(
            "K @ u", size=matrix_shape[0], calc=self.calc_output
        )

    def get_matrix_input_port(self) -> InputPort:
        return self.get_input_port(0)

    def get_vector_input_port(self) -> InputPort:
        return self.get_input_port(1)

    def calc_output(self, context: Context, output: BasicVector):
        K = np.asarray(
            self.get_matrix_input_port().Eval(context), dtype=np.double  # noqa
        )
        u = np.asarray(
            self.get_vector_input_port().Eval(context), dtype=np.double  # noqa
        )

        output.set_value(K @ u)
