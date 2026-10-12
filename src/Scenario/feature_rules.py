
FEATURE_RULES = {
    "Flow_Duration": {
        "method": "mean",
        "description": "Mean flow duration.",
    },

    "Total_Fwd_Packets": {
        "method": "sum",
        "description": "Sum of forward packets.",
    },
    "Total_Backward_Packets": {
        "method": "sum",
        "description": "Sum of backward packets.",
    },
    "Total_Length_of_Fwd_Packets": {
        "method": "sum",
        "description": "Sum of forward bytes.",
    },
    "Total_Length_of_Bwd_Packets": {
        "method": "sum",
        "description": "Sum of backward bytes.",
    },

    "Fwd_Packet_Length_Max": {
        "method": "max",
        "description": "Maximum forward packet length.",
    },
    "Fwd_Packet_Length_Min": {
        "method": "min",
        "description": "Minimum forward packet length.",
    },
    "Fwd_Packet_Length_Mean": {
        "method": "weighted_mean",
        "weight": "Total_Fwd_Packets",
        "description": "Weighted mean by forward packet count.",
    },
    "Fwd_Packet_Length_Std": {
        "method": "pooled_std",
        "count": "Total_Fwd_Packets",
        "mean": "Fwd_Packet_Length_Mean",
        "description": "Pooled standard deviation of forward packets.",
    },

    "Bwd_Packet_Length_Max": {
        "method": "max",
        "description": "Maximum backward packet length.",
    },
    "Bwd_Packet_Length_Min": {
        "method": "min",
        "description": "Minimum backward packet length.",
    },
    "Bwd_Packet_Length_Mean": {
        "method": "weighted_mean",
        "weight": "Total_Backward_Packets",
        "description": "Weighted mean by backward packet count.",
    },
    "Bwd_Packet_Length_Std": {
        "method": "pooled_std",
        "count": "Total_Backward_Packets",
        "mean": "Bwd_Packet_Length_Mean",
        "description": "Pooled standard deviation of backward packets.",
    },

    "Flow_Bytes_s": {
        "method": "weighted_mean",
        "weight": "Flow_Duration",
        "description": "Flow Bytes/s weighted mean by flow duration.",
    },
    "Flow_Packets_s": {
        "method": "weighted_mean",
        "weight": "Flow_Duration",
        "description": "Flow Packets/s weighted mean by flow duration.",
    },

    "Flow_IAT_Mean": {
        "method": "weighted_mean",
        "weight": "__FlowIATCount",
        "description": "Weighted mean by estimated Flow IAT count.",
    },
    "Flow_IAT_Std": {
        "method": "pooled_std",
        "count": "__FlowIATCount",
        "mean": "Flow_IAT_Mean",
        "description": "Pooled standard deviation of Flow IATs.",
    },
    "Flow_IAT_Max": {
        "method": "max",
        "description": "Maximum Flow IAT.",
    },
    "Flow_IAT_Min": {
        "method": "min",
        "description": "Minimum Flow IAT.",
    },

    "Fwd_IAT_Total": {
        "method": "sum",
        "description": "Sum of Fwd IAT Total.",
    },
    "Fwd_IAT_Mean": {
        "method": "weighted_mean",
        "weight": "__FwdIATCount",
        "description": "Weighted mean by estimated forward intervals.",
    },
    "Fwd_IAT_Std": {
        "method": "pooled_std",
        "count": "__FwdIATCount",
        "mean": "Fwd_IAT_Mean",
        "description": "Pooled standard deviation of Fwd IAT.",
    },
    "Fwd_IAT_Max": {
        "method": "max",
        "description": "Maximum Fwd IAT.",
    },
    "Fwd_IAT_Min": {
        "method": "min",
        "description": "Minimum Fwd IAT.",
    },

    "Bwd_IAT_Total": {
        "method": "sum",
        "description": "Sum of Bwd IAT Total.",
    },
    "Bwd_IAT_Mean": {
        "method": "weighted_mean",
        "weight": "__BwdIATCount",
        "description": "Weighted mean by estimated backward intervals.",
    },
    "Bwd_IAT_Std": {
        "method": "pooled_std",
        "count": "__BwdIATCount",
        "mean": "Bwd_IAT_Mean",
        "description": "Pooled standard deviation of Bwd IAT.",
    },
    "Bwd_IAT_Max": {
        "method": "max",
        "description": "Maximum Bwd IAT.",
    },
    "Bwd_IAT_Min": {
        "method": "min",
        "description": "Minimum Bwd IAT.",
    },

    "Fwd_PSH_Flags": {
        "method": "sum",
        "description": "Sum of Fwd PSH Flags.",
    },
    "Bwd_PSH_Flags": {
        "method": "sum",
        "description": "Sum of Bwd PSH Flags.",
    },
    "Fwd_URG_Flags": {
        "method": "sum",
        "description": "Sum of Fwd URG Flags.",
    },
    "Bwd_URG_Flags": {
        "method": "sum",
        "description": "Sum of Bwd URG Flags.",
    },
    "Fwd_Header_Length": {
        "method": "sum",
        "description": "Sum of Fwd Header Length.",
    },
    "Bwd_Header_Length": {
        "method": "sum",
        "description": "Sum of Bwd Header Length.",
    },

    "Fwd_Packets_s": {
        "method": "weighted_mean",
        "weight": "Flow_Duration",
        "description": "Fwd Packets/s weighted mean by flow duration.",
    },
    "Bwd_Packets_s": {
        "method": "weighted_mean",
        "weight": "Flow_Duration",
        "description": "Bwd Packets/s weighted mean by flow duration.",
    },

    "Min_Packet_Length": {
        "method": "min",
        "description": "Minimum packet length.",
    },
    "Max_Packet_Length": {
        "method": "max",
        "description": "Maximum packet length.",
    },
    "Packet_Length_Mean": {
        "method": "weighted_mean",
        "weight": "__TotalPackets",
        "description": "Weighted mean by total packet count.",
    },
    "Packet_Length_Std": {
        "method": "pooled_std",
        "count": "__TotalPackets",
        "mean": "Packet_Length_Mean",
        "description": "Pooled standard deviation of packet length.",
    },
    "Packet_Length_Variance": {
        "method": "variance",
        "std": "Packet_Length_Std",
        "description": "Variance derived from the pooled standard deviation.",
    },

    "FIN_Flag_Count": {
        "method": "sum",
        "description": "Sum of FIN flags.",
    },
    "SYN_Flag_Count": {
        "method": "sum",
        "description": "Sum of SYN flags.",
    },
    "RST_Flag_Count": {
        "method": "sum",
        "description": "Sum of RST flags.",
    },
    "PSH_Flag_Count": {
        "method": "sum",
        "description": "Sum of PSH flags.",
    },
    "ACK_Flag_Count": {
        "method": "sum",
        "description": "Sum of ACK flags.",
    },
    "URG_Flag_Count": {
        "method": "sum",
        "description": "Sum of URG flags.",
    },
    "CWE_Flag_Count": {
        "method": "sum",
        "description": "Sum of CWE flags.",
    },
    "ECE_Flag_Count": {
        "method": "sum",
        "description": "Sum of ECE flags.",
    },

    "Down_Up_Ratio": {
        "method": "mean",
        "description": "Mean Down/Up Ratio across flows.",
    },
    "Average_Packet_Size": {
        "method": "weighted_mean",
        "weight": "__TotalPackets",
        "description": "Weighted mean by total packet count.",
    },
    "Avg_Fwd_Segment_Size": {
        "method": "weighted_mean",
        "weight": "Total_Fwd_Packets",
        "description": "Weighted mean by forward packet count.",
    },
    "Avg_Bwd_Segment_Size": {
        "method": "weighted_mean",
        "weight": "Total_Backward_Packets",
        "description": "Weighted mean by backward packet count.",
    },

    "Fwd_Avg_Bytes_Bulk": {
        "method": "mean",
        "description": "Mean across flows.",
    },
    "Fwd_Avg_Packets_Bulk": {
        "method": "mean",
        "description": "Mean across flows.",
    },
    "Fwd_Avg_Bulk_Rate": {
        "method": "mean",
        "description": "Mean across flows.",
    },
    "Bwd_Avg_Bytes_Bulk": {
        "method": "mean",
        "description": "Mean across flows.",
    },
    "Bwd_Avg_Packets_Bulk": {
        "method": "mean",
        "description": "Mean across flows.",
    },
    "Bwd_Avg_Bulk_Rate": {
        "method": "mean",
        "description": "Mean across flows.",
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
        "description": "Mean initial forward TCP window.",
    },
    "Init_Win_bytes_backward": {
        "method": "mean",
        "description": "Mean initial backward TCP window.",
    },
    "act_data_pkt_fwd": {
        "method": "sum",
        "description": "Soma dos active data packets forward.",
    },
    "min_seg_size_forward": {
        "method": "min",
        "description": "Minimum forward segment size.",
    },

    "Active_Mean": {
        "method": "mean",
        "description": "Mean of Active Mean across flows.",
    },
    "Active_Std": {
        "method": "mean",
        "description": "Mean of Active Std across flows.",
    },
    "Active_Max": {
        "method": "max",
        "description": "Maximum Active Max.",
    },
    "Active_Min": {
        "method": "min",
        "description": "Minimum Active Min.",
    },

    "Idle_Mean": {
        "method": "mean",
        "description": "Mean of Idle Mean across flows.",
    },
    "Idle_Std": {
        "method": "mean",
        "description": "Mean of Idle Std across flows.",
    },
    "Idle_Max": {
        "method": "max",
        "description": "Maximum Idle Max.",
    },
    "Idle_Min": {
        "method": "min",
        "description": "Minimum Idle Min.",
    },
}
