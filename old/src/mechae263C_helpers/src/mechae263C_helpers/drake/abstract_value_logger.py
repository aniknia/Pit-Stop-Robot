import copy

from pydrake.common.value import AbstractValue
from pydrake.systems.framework import LeafSystem, OutputPort, DiagramBuilder


class AbstractValueLogger(LeafSystem):
    def __init__(self, output_port_to_log: OutputPort, builder: DiagramBuilder):
        super().__init__()
        output_port_to_log.get_data_type()
        self.DeclareAbstractInputPort(
            "input", AbstractValue.Make(output_port_to_log.get_data_type())
        )
        self.times = []
        self.values = []

    def publish(self, context):
        self.times.append(context.get_time())
        self.values.append(copy.deepcopy(self.get_input_port(0).Eval(context)))
