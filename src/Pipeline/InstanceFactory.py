import numpy as np


class InstanceFactory:
    @staticmethod
    def createLabeledInstance(raw_instance, values):
        # Reconstrói uma instância rotulada preservando o schema e o rótulo binário original.
        from capymoa.core import LabeledInstance
        return LabeledInstance.from_array(
            schema=raw_instance.schema,
            x=np.asarray(values, dtype=np.float64),
            y_index=int(raw_instance.y_index),
        )

    @staticmethod
    def getStreamSchema(stream):
        # Obtém o schema da stream sem depender de uma única versão da API do CapyMOA.
        if hasattr(stream, "get_schema"):
            return stream.get_schema()
        if hasattr(stream, "schema"):
            return stream.schema
        raise AttributeError("Unable to obtain the stream schema.")
