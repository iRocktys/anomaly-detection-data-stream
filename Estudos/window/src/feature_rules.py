# Regras explícitas para as features do CICFlowMeter.
#
# Source_IP, Destination_IP, Source_Port, Destination_Port e Protocol
# NÃO são exportados para o CSV final. Eles são usados somente para
# o cálculo das entropias de Shannon em entropy.py.
#
# Métodos:
# sum            soma
# mean           média aritmética
# min            mínimo
# max            máximo
# weighted_mean  média ponderada
# pooled_std     desvio-padrão combinado
# variance       quadrado do pooled_std correspondente

FEATURE_RULES = {
    "Flow_Duration": {
        "method": "mean",
        "description": "Média das durações dos fluxos.",
    },

    "Total_Fwd_Packets": {
        "method": "sum",
        "description": "Soma dos pacotes forward.",
    },
    "Total_Backward_Packets": {
        "method": "sum",
        "description": "Soma dos pacotes backward.",
    },
    "Total_Length_of_Fwd_Packets": {
        "method": "sum",
        "description": "Soma dos bytes forward.",
    },
    "Total_Length_of_Bwd_Packets": {
        "method": "sum",
        "description": "Soma dos bytes backward.",
    },

    "Fwd_Packet_Length_Max": {
        "method": "max",
        "description": "Maior comprimento de pacote forward.",
    },
    "Fwd_Packet_Length_Min": {
        "method": "min",
        "description": "Menor comprimento de pacote forward.",
    },
    "Fwd_Packet_Length_Mean": {
        "method": "weighted_mean",
        "weight": "Total_Fwd_Packets",
        "description": "Média ponderada pela quantidade de pacotes forward.",
    },
    "Fwd_Packet_Length_Std": {
        "method": "pooled_std",
        "count": "Total_Fwd_Packets",
        "mean": "Fwd_Packet_Length_Mean",
        "description": "Desvio-padrão combinado dos pacotes forward.",
    },

    "Bwd_Packet_Length_Max": {
        "method": "max",
        "description": "Maior comprimento de pacote backward.",
    },
    "Bwd_Packet_Length_Min": {
        "method": "min",
        "description": "Menor comprimento de pacote backward.",
    },
    "Bwd_Packet_Length_Mean": {
        "method": "weighted_mean",
        "weight": "Total_Backward_Packets",
        "description": "Média ponderada pela quantidade de pacotes backward.",
    },
    "Bwd_Packet_Length_Std": {
        "method": "pooled_std",
        "count": "Total_Backward_Packets",
        "mean": "Bwd_Packet_Length_Mean",
        "description": "Desvio-padrão combinado dos pacotes backward.",
    },

    "Flow_Bytes_s": {
        "method": "weighted_mean",
        "weight": "Flow_Duration",
        "description": "Média ponderada de Flow Bytes/s pela duração do fluxo.",
    },
    "Flow_Packets_s": {
        "method": "weighted_mean",
        "weight": "Flow_Duration",
        "description": "Média ponderada de Flow Packets/s pela duração do fluxo.",
    },

    "Flow_IAT_Mean": {
        "method": "weighted_mean",
        "weight": "__FlowIATCount",
        "description": "Média ponderada pela quantidade estimada de Flow IATs.",
    },
    "Flow_IAT_Std": {
        "method": "pooled_std",
        "count": "__FlowIATCount",
        "mean": "Flow_IAT_Mean",
        "description": "Desvio-padrão combinado dos Flow IATs.",
    },
    "Flow_IAT_Max": {
        "method": "max",
        "description": "Maior Flow IAT.",
    },
    "Flow_IAT_Min": {
        "method": "min",
        "description": "Menor Flow IAT.",
    },

    "Fwd_IAT_Total": {
        "method": "sum",
        "description": "Soma do Fwd IAT Total.",
    },
    "Fwd_IAT_Mean": {
        "method": "weighted_mean",
        "weight": "__FwdIATCount",
        "description": "Média ponderada pelos intervalos forward estimados.",
    },
    "Fwd_IAT_Std": {
        "method": "pooled_std",
        "count": "__FwdIATCount",
        "mean": "Fwd_IAT_Mean",
        "description": "Desvio-padrão combinado do Fwd IAT.",
    },
    "Fwd_IAT_Max": {
        "method": "max",
        "description": "Maior Fwd IAT.",
    },
    "Fwd_IAT_Min": {
        "method": "min",
        "description": "Menor Fwd IAT.",
    },

    "Bwd_IAT_Total": {
        "method": "sum",
        "description": "Soma do Bwd IAT Total.",
    },
    "Bwd_IAT_Mean": {
        "method": "weighted_mean",
        "weight": "__BwdIATCount",
        "description": "Média ponderada pelos intervalos backward estimados.",
    },
    "Bwd_IAT_Std": {
        "method": "pooled_std",
        "count": "__BwdIATCount",
        "mean": "Bwd_IAT_Mean",
        "description": "Desvio-padrão combinado do Bwd IAT.",
    },
    "Bwd_IAT_Max": {
        "method": "max",
        "description": "Maior Bwd IAT.",
    },
    "Bwd_IAT_Min": {
        "method": "min",
        "description": "Menor Bwd IAT.",
    },

    "Fwd_PSH_Flags": {
        "method": "sum",
        "description": "Soma das Fwd PSH Flags.",
    },
    "Bwd_PSH_Flags": {
        "method": "sum",
        "description": "Soma das Bwd PSH Flags.",
    },
    "Fwd_URG_Flags": {
        "method": "sum",
        "description": "Soma das Fwd URG Flags.",
    },
    "Bwd_URG_Flags": {
        "method": "sum",
        "description": "Soma das Bwd URG Flags.",
    },
    "Fwd_Header_Length": {
        "method": "sum",
        "description": "Soma do Fwd Header Length.",
    },
    "Bwd_Header_Length": {
        "method": "sum",
        "description": "Soma do Bwd Header Length.",
    },

    "Fwd_Packets_s": {
        "method": "weighted_mean",
        "weight": "Flow_Duration",
        "description": "Média ponderada de Fwd Packets/s pela duração.",
    },
    "Bwd_Packets_s": {
        "method": "weighted_mean",
        "weight": "Flow_Duration",
        "description": "Média ponderada de Bwd Packets/s pela duração.",
    },

    "Min_Packet_Length": {
        "method": "min",
        "description": "Menor comprimento de pacote.",
    },
    "Max_Packet_Length": {
        "method": "max",
        "description": "Maior comprimento de pacote.",
    },
    "Packet_Length_Mean": {
        "method": "weighted_mean",
        "weight": "__TotalPackets",
        "description": "Média ponderada pelo total de pacotes.",
    },
    "Packet_Length_Std": {
        "method": "pooled_std",
        "count": "__TotalPackets",
        "mean": "Packet_Length_Mean",
        "description": "Desvio-padrão combinado do comprimento de pacote.",
    },
    "Packet_Length_Variance": {
        "method": "variance",
        "std": "Packet_Length_Std",
        "description": "Variância derivada do desvio-padrão combinado.",
    },

    "FIN_Flag_Count": {
        "method": "sum",
        "description": "Soma das FIN flags.",
    },
    "SYN_Flag_Count": {
        "method": "sum",
        "description": "Soma das SYN flags.",
    },
    "RST_Flag_Count": {
        "method": "sum",
        "description": "Soma das RST flags.",
    },
    "PSH_Flag_Count": {
        "method": "sum",
        "description": "Soma das PSH flags.",
    },
    "ACK_Flag_Count": {
        "method": "sum",
        "description": "Soma das ACK flags.",
    },
    "URG_Flag_Count": {
        "method": "sum",
        "description": "Soma das URG flags.",
    },
    "CWE_Flag_Count": {
        "method": "sum",
        "description": "Soma das CWE flags.",
    },
    "ECE_Flag_Count": {
        "method": "sum",
        "description": "Soma das ECE flags.",
    },

    "Down_Up_Ratio": {
        "method": "mean",
        "description": "Média do Down/Up Ratio dos fluxos.",
    },
    "Average_Packet_Size": {
        "method": "weighted_mean",
        "weight": "__TotalPackets",
        "description": "Média ponderada pelo total de pacotes.",
    },
    "Avg_Fwd_Segment_Size": {
        "method": "weighted_mean",
        "weight": "Total_Fwd_Packets",
        "description": "Média ponderada pelos pacotes forward.",
    },
    "Avg_Bwd_Segment_Size": {
        "method": "weighted_mean",
        "weight": "Total_Backward_Packets",
        "description": "Média ponderada pelos pacotes backward.",
    },

    "Fwd_Avg_Bytes_Bulk": {
        "method": "mean",
        "description": "Média entre os fluxos.",
    },
    "Fwd_Avg_Packets_Bulk": {
        "method": "mean",
        "description": "Média entre os fluxos.",
    },
    "Fwd_Avg_Bulk_Rate": {
        "method": "mean",
        "description": "Média entre os fluxos.",
    },
    "Bwd_Avg_Bytes_Bulk": {
        "method": "mean",
        "description": "Média entre os fluxos.",
    },
    "Bwd_Avg_Packets_Bulk": {
        "method": "mean",
        "description": "Média entre os fluxos.",
    },
    "Bwd_Avg_Bulk_Rate": {
        "method": "mean",
        "description": "Média entre os fluxos.",
    },

    "Subflow_Fwd_Packets": {
        "method": "sum",
        "description": "Soma dos Subflow Fwd Packets.",
    },
    "Subflow_Fwd_Bytes": {
        "method": "sum",
        "description": "Soma dos Subflow Fwd Bytes.",
    },
    "Subflow_Bwd_Packets": {
        "method": "sum",
        "description": "Soma dos Subflow Bwd Packets.",
    },
    "Subflow_Bwd_Bytes": {
        "method": "sum",
        "description": "Soma dos Subflow Bwd Bytes.",
    },

    "Init_Win_bytes_forward": {
        "method": "mean",
        "description": "Média da janela TCP inicial forward.",
    },
    "Init_Win_bytes_backward": {
        "method": "mean",
        "description": "Média da janela TCP inicial backward.",
    },
    "act_data_pkt_fwd": {
        "method": "sum",
        "description": "Soma dos active data packets forward.",
    },
    "min_seg_size_forward": {
        "method": "min",
        "description": "Menor min segment size forward.",
    },

    "Active_Mean": {
        "method": "mean",
        "description": "Média dos Active Mean dos fluxos.",
    },
    "Active_Std": {
        "method": "mean",
        "description": "Média dos Active Std dos fluxos.",
    },
    "Active_Max": {
        "method": "max",
        "description": "Maior Active Max.",
    },
    "Active_Min": {
        "method": "min",
        "description": "Menor Active Min.",
    },

    "Idle_Mean": {
        "method": "mean",
        "description": "Média dos Idle Mean dos fluxos.",
    },
    "Idle_Std": {
        "method": "mean",
        "description": "Média dos Idle Std dos fluxos.",
    },
    "Idle_Max": {
        "method": "max",
        "description": "Maior Idle Max.",
    },
    "Idle_Min": {
        "method": "min",
        "description": "Menor Idle Min.",
    },
}
