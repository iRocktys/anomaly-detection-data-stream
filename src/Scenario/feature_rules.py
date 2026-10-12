
FEATURE_RULES = {
    "Flow_Duration": {
        "method": "mean",
        "description": "Mean aggregation for Flow_Duration.",
    },

    "Total_Fwd_Packets": {
        "method": "sum",
        "description": "Sum aggregation for Total_Fwd_Packets.",
    },
    "Total_Backward_Packets": {
        "method": "sum",
        "description": "Sum aggregation for Total_Backward_Packets.",
    },
    "Total_Length_of_Fwd_Packets": {
        "method": "sum",
        "description": "Sum aggregation for Total_Length_of_Fwd_Packets.",
    },
    "Total_Length_of_Bwd_Packets": {
        "method": "sum",
        "description": "Sum aggregation for Total_Length_of_Bwd_Packets.",
    },

    "Fwd_Packet_Length_Max": {
        "method": "max",
        "description": "Maximum aggregation for Fwd_Packet_Length_Max.",
    },
    "Fwd_Packet_Length_Min": {
        "method": "min",
        "description": "Minimum aggregation for Fwd_Packet_Length_Min.",
    },
    "Fwd_Packet_Length_Mean": {
        "method": "weighted_mean",
        "weight": "Total_Fwd_Packets",
        "description": "Weighted mean aggregation for Fwd_Packet_Length_Mean.",
    },
    "Fwd_Packet_Length_Std": {
        "method": "pooled_std",
        "count": "Total_Fwd_Packets",
        "mean": "Fwd_Packet_Length_Mean",
        "description": "Pooled standard deviation aggregation for Fwd_Packet_Length_Std.",
    },

    "Bwd_Packet_Length_Max": {
        "method": "max",
        "description": "Maximum aggregation for Bwd_Packet_Length_Max.",
    },
    "Bwd_Packet_Length_Min": {
        "method": "min",
        "description": "Minimum aggregation for Bwd_Packet_Length_Min.",
    },
    "Bwd_Packet_Length_Mean": {
        "method": "weighted_mean",
        "weight": "Total_Backward_Packets",
        "description": "Weighted mean aggregation for Bwd_Packet_Length_Mean.",
    },
    "Bwd_Packet_Length_Std": {
        "method": "pooled_std",
        "count": "Total_Backward_Packets",
        "mean": "Bwd_Packet_Length_Mean",
        "description": "Pooled standard deviation aggregation for Bwd_Packet_Length_Std.",
    },

    "Flow_Bytes_s": {
        "method": "weighted_mean",
        "weight": "Flow_Duration",
        "description": "Weighted mean aggregation for Flow_Bytes_s.",
    },
    "Flow_Packets_s": {
        "method": "weighted_mean",
        "weight": "Flow_Duration",
        "description": "Weighted mean aggregation for Flow_Packets_s.",
    },

    "Flow_IAT_Mean": {
        "method": "weighted_mean",
        "weight": "__FlowIATCount",
        "description": "Weighted mean aggregation for Flow_IAT_Mean.",
    },
    "Flow_IAT_Std": {
        "method": "pooled_std",
        "count": "__FlowIATCount",
        "mean": "Flow_IAT_Mean",
        "description": "Pooled standard deviation aggregation for Flow_IAT_Std.",
    },
    "Flow_IAT_Max": {
        "method": "max",
        "description": "Maximum aggregation for Flow_IAT_Max.",
    },
    "Flow_IAT_Min": {
        "method": "min",
        "description": "Minimum aggregation for Flow_IAT_Min.",
    },

    "Fwd_IAT_Total": {
        "method": "sum",
        "description": "Sum aggregation for Fwd_IAT_Total.",
    },
    "Fwd_IAT_Mean": {
        "method": "weighted_mean",
        "weight": "__FwdIATCount",
        "description": "Weighted mean aggregation for Fwd_IAT_Mean.",
    },
    "Fwd_IAT_Std": {
        "method": "pooled_std",
        "count": "__FwdIATCount",
        "mean": "Fwd_IAT_Mean",
        "description": "Pooled standard deviation aggregation for Fwd_IAT_Std.",
    },
    "Fwd_IAT_Max": {
        "method": "max",
        "description": "Maximum aggregation for Fwd_IAT_Max.",
    },
    "Fwd_IAT_Min": {
        "method": "min",
        "description": "Minimum aggregation for Fwd_IAT_Min.",
    },

    "Bwd_IAT_Total": {
        "method": "sum",
        "description": "Sum aggregation for Bwd_IAT_Total.",
    },
    "Bwd_IAT_Mean": {
        "method": "weighted_mean",
        "weight": "__BwdIATCount",
        "description": "Weighted mean aggregation for Bwd_IAT_Mean.",
    },
    "Bwd_IAT_Std": {
        "method": "pooled_std",
        "count": "__BwdIATCount",
        "mean": "Bwd_IAT_Mean",
        "description": "Pooled standard deviation aggregation for Bwd_IAT_Std.",
    },
    "Bwd_IAT_Max": {
        "method": "max",
        "description": "Maximum aggregation for Bwd_IAT_Max.",
    },
    "Bwd_IAT_Min": {
        "method": "min",
        "description": "Minimum aggregation for Bwd_IAT_Min.",
    },

    "Fwd_PSH_Flags": {
        "method": "sum",
        "description": "Sum aggregation for Fwd_PSH_Flags.",
    },
    "Bwd_PSH_Flags": {
        "method": "sum",
        "description": "Sum aggregation for Bwd_PSH_Flags.",
    },
    "Fwd_URG_Flags": {
        "method": "sum",
        "description": "Sum aggregation for Fwd_URG_Flags.",
    },
    "Bwd_URG_Flags": {
        "method": "sum",
        "description": "Sum aggregation for Bwd_URG_Flags.",
    },
    "Fwd_Header_Length": {
        "method": "sum",
        "description": "Sum aggregation for Fwd_Header_Length.",
    },
    "Bwd_Header_Length": {
        "method": "sum",
        "description": "Sum aggregation for Bwd_Header_Length.",
    },

    "Fwd_Packets_s": {
        "method": "weighted_mean",
        "weight": "Flow_Duration",
        "description": "Weighted mean aggregation for Fwd_Packets_s.",
    },
    "Bwd_Packets_s": {
        "method": "weighted_mean",
        "weight": "Flow_Duration",
        "description": "Weighted mean aggregation for Bwd_Packets_s.",
    },

    "Min_Packet_Length": {
        "method": "min",
        "description": "Minimum aggregation for Min_Packet_Length.",
    },
    "Max_Packet_Length": {
        "method": "max",
        "description": "Maximum aggregation for Max_Packet_Length.",
    },
    "Packet_Length_Mean": {
        "method": "weighted_mean",
        "weight": "__TotalPackets",
        "description": "Weighted mean aggregation for Packet_Length_Mean.",
    },
    "Packet_Length_Std": {
        "method": "pooled_std",
        "count": "__TotalPackets",
        "mean": "Packet_Length_Mean",
        "description": "Pooled standard deviation aggregation for Packet_Length_Std.",
    },
    "Packet_Length_Variance": {
        "method": "variance",
        "std": "Packet_Length_Std",
        "description": "Variance derived from the configured standard deviation for Packet_Length_Variance.",
    },

    "FIN_Flag_Count": {
        "method": "sum",
        "description": "Sum aggregation for FIN_Flag_Count.",
    },
    "SYN_Flag_Count": {
        "method": "sum",
        "description": "Sum aggregation for SYN_Flag_Count.",
    },
    "RST_Flag_Count": {
        "method": "sum",
        "description": "Sum aggregation for RST_Flag_Count.",
    },
    "PSH_Flag_Count": {
        "method": "sum",
        "description": "Sum aggregation for PSH_Flag_Count.",
    },
    "ACK_Flag_Count": {
        "method": "sum",
        "description": "Sum aggregation for ACK_Flag_Count.",
    },
    "URG_Flag_Count": {
        "method": "sum",
        "description": "Sum aggregation for URG_Flag_Count.",
    },
    "CWE_Flag_Count": {
        "method": "sum",
        "description": "Sum aggregation for CWE_Flag_Count.",
    },
    "ECE_Flag_Count": {
        "method": "sum",
        "description": "Sum aggregation for ECE_Flag_Count.",
    },

    "Down_Up_Ratio": {
        "method": "mean",
        "description": "Mean aggregation for Down_Up_Ratio.",
    },
    "Average_Packet_Size": {
        "method": "weighted_mean",
        "weight": "__TotalPackets",
        "description": "Weighted mean aggregation for Average_Packet_Size.",
    },
    "Avg_Fwd_Segment_Size": {
        "method": "weighted_mean",
        "weight": "Total_Fwd_Packets",
        "description": "Weighted mean aggregation for Avg_Fwd_Segment_Size.",
    },
    "Avg_Bwd_Segment_Size": {
        "method": "weighted_mean",
        "weight": "Total_Backward_Packets",
        "description": "Weighted mean aggregation for Avg_Bwd_Segment_Size.",
    },

    "Fwd_Avg_Bytes_Bulk": {
        "method": "mean",
        "description": "Mean aggregation for Fwd_Avg_Bytes_Bulk.",
    },
    "Fwd_Avg_Packets_Bulk": {
        "method": "mean",
        "description": "Mean aggregation for Fwd_Avg_Packets_Bulk.",
    },
    "Fwd_Avg_Bulk_Rate": {
        "method": "mean",
        "description": "Mean aggregation for Fwd_Avg_Bulk_Rate.",
    },
    "Bwd_Avg_Bytes_Bulk": {
        "method": "mean",
        "description": "Mean aggregation for Bwd_Avg_Bytes_Bulk.",
    },
    "Bwd_Avg_Packets_Bulk": {
        "method": "mean",
        "description": "Mean aggregation for Bwd_Avg_Packets_Bulk.",
    },
    "Bwd_Avg_Bulk_Rate": {
        "method": "mean",
        "description": "Mean aggregation for Bwd_Avg_Bulk_Rate.",
    },

    "Subflow_Fwd_Packets": {
        "method": "sum",
        "description": "Sum aggregation for Subflow_Fwd_Packets.",
    },
    "Subflow_Fwd_Bytes": {
        "method": "sum",
        "description": "Sum aggregation for Subflow_Fwd_Bytes.",
    },
    "Subflow_Bwd_Packets": {
        "method": "sum",
        "description": "Sum aggregation for Subflow_Bwd_Packets.",
    },
    "Subflow_Bwd_Bytes": {
        "method": "sum",
        "description": "Sum aggregation for Subflow_Bwd_Bytes.",
    },

    "Init_Win_bytes_forward": {
        "method": "mean",
        "description": "Mean aggregation for Init_Win_bytes_forward.",
    },
    "Init_Win_bytes_backward": {
        "method": "mean",
        "description": "Mean aggregation for Init_Win_bytes_backward.",
    },
    "act_data_pkt_fwd": {
        "method": "sum",
        "description": "Sum aggregation for act_data_pkt_fwd.",
    },
    "min_seg_size_forward": {
        "method": "min",
        "description": "Minimum aggregation for min_seg_size_forward.",
    },

    "Active_Mean": {
        "method": "mean",
        "description": "Mean aggregation for Active_Mean.",
    },
    "Active_Std": {
        "method": "mean",
        "description": "Mean aggregation for Active_Std.",
    },
    "Active_Max": {
        "method": "max",
        "description": "Maximum aggregation for Active_Max.",
    },
    "Active_Min": {
        "method": "min",
        "description": "Minimum aggregation for Active_Min.",
    },

    "Idle_Mean": {
        "method": "mean",
        "description": "Mean aggregation for Idle_Mean.",
    },
    "Idle_Std": {
        "method": "mean",
        "description": "Mean aggregation for Idle_Std.",
    },
    "Idle_Max": {
        "method": "max",
        "description": "Maximum aggregation for Idle_Max.",
    },
    "Idle_Min": {
        "method": "min",
        "description": "Minimum aggregation for Idle_Min.",
    },
}
