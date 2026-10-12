import numpy as np


class InstanceFactory:
    @staticmethod
    def createLabeledInstance(rawInstance, values):
        # Reconstrói uma instância rotulada compatível com versões antigas e novas do CapyMOA.
        try:
            from capymoa.instance import LabeledInstance
        except ImportError:
            from capymoa.core import LabeledInstance
        return LabeledInstance.from_array(
            schema=rawInstance.schema,
            x=np.asarray(values, dtype=np.float64),
            y_index=int(rawInstance.y_index),
        )

    @staticmethod
    def getStreamSchema(stream):
        # Obtém o schema da stream pelas interfaces suportadas pelo CapyMOA.
        if hasattr(stream, "get_schema"):
            return stream.get_schema()
        if hasattr(stream, "schema"):
            return stream.schema
        raise AttributeError("The stream does not expose a schema.")
