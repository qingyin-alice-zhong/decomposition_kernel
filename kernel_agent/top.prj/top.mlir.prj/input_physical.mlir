module {
  aie.device(npu1_4col) {
    memref.global "public" @fifo_23_cons : memref<1x1x1x512xf32>
    memref.global "public" @fifo_23 : memref<1x1x1x512xf32>
    memref.global "public" @fifo_22_cons : memref<512xf32>
    memref.global "public" @fifo_22 : memref<512xf32>
    memref.global "public" @fifo_21_cons : memref<1x1x1x300xf32>
    memref.global "public" @fifo_21 : memref<1x1x1x300xf32>
    memref.global "public" @fifo_20_cons : memref<300xf32>
    memref.global "public" @fifo_20 : memref<300xf32>
    memref.global "public" @fifo_19_cons : memref<1x1x1x224xf32>
    memref.global "public" @fifo_19 : memref<1x1x1x224xf32>
    memref.global "public" @fifo_18_cons : memref<224xf32>
    memref.global "public" @fifo_18 : memref<224xf32>
    memref.global "public" @fifo_17_cons : memref<1x1x1x512xf32>
    memref.global "public" @fifo_17 : memref<1x1x1x512xf32>
    memref.global "public" @fifo_16_cons : memref<512xf32>
    memref.global "public" @fifo_16 : memref<512xf32>
    memref.global "public" @fifo_15_cons : memref<1x1x1x224xf32>
    memref.global "public" @fifo_15 : memref<1x1x1x224xf32>
    memref.global "public" @fifo_14_cons : memref<224xf32>
    memref.global "public" @fifo_14 : memref<224xf32>
    memref.global "public" @fifo_13_cons : memref<1x1x1x300xf32>
    memref.global "public" @fifo_13 : memref<1x1x1x300xf32>
    memref.global "public" @fifo_12_cons : memref<300xf32>
    memref.global "public" @fifo_12 : memref<300xf32>
    memref.global "public" @fifo_11_cons : memref<1x1x1x512xf32>
    memref.global "public" @fifo_11 : memref<1x1x1x512xf32>
    memref.global "public" @fifo_10_cons : memref<512xf32>
    memref.global "public" @fifo_10 : memref<512xf32>
    memref.global "public" @fifo_9_cons : memref<1x1x1x224xf32>
    memref.global "public" @fifo_9 : memref<1x1x1x224xf32>
    memref.global "public" @fifo_8_cons : memref<224xf32>
    memref.global "public" @fifo_8 : memref<224xf32>
    memref.global "public" @fifo_7_cons : memref<1x1x1x300xf32>
    memref.global "public" @fifo_7 : memref<1x1x1x300xf32>
    memref.global "public" @fifo_6_cons : memref<300xf32>
    memref.global "public" @fifo_6 : memref<300xf32>
    memref.global "public" @fifo_5_cons : memref<1x1x1x512xf32>
    memref.global "public" @fifo_5 : memref<1x1x1x512xf32>
    memref.global "public" @fifo_4_cons : memref<512xf32>
    memref.global "public" @fifo_4 : memref<512xf32>
    memref.global "public" @fifo_3_cons : memref<1x1x1x224xf32>
    memref.global "public" @fifo_3 : memref<1x1x1x224xf32>
    memref.global "public" @fifo_2_cons : memref<224xf32>
    memref.global "public" @fifo_2 : memref<224xf32>
    memref.global "public" @fifo_1_cons : memref<1x1x1x300xf32>
    memref.global "public" @fifo_1 : memref<1x1x1x300xf32>
    memref.global "public" @fifo_0_cons : memref<300xf32>
    memref.global "public" @fifo_0 : memref<300xf32>
    func.func private @conv2d_3x64_b1a_fp32(memref<300xf32>, memref<512xf32>, memref<224xf32>)
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_3_1 = aie.tile(3, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %fifo_23_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 4) {init = 1 : i32, sym_name = "fifo_23_cons_prod_lock_0"}
    %fifo_23_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 5) {init = 0 : i32, sym_name = "fifo_23_cons_cons_lock_0"}
    %fifo_22_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, sym_name = "fifo_22_cons_buff_0"} : memref<512xf32> 
    %fifo_22_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 2048 : i32, sym_name = "fifo_22_cons_buff_1"} : memref<512xf32> 
    %fifo_22_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 6) {init = 2 : i32, sym_name = "fifo_22_cons_prod_lock_0"}
    %fifo_22_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 7) {init = 0 : i32, sym_name = "fifo_22_cons_cons_lock_0"}
    %fifo_22_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, sym_name = "fifo_22_buff_0"} : memref<512xf32> 
    %fifo_22_buff_1 = aie.buffer(%tile_1_2) {address = 3072 : i32, sym_name = "fifo_22_buff_1"} : memref<512xf32> 
    %fifo_22_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 2 : i32, sym_name = "fifo_22_prod_lock_0"}
    %fifo_22_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "fifo_22_cons_lock_0"}
    %fifo_21_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 4096 : i32, sym_name = "fifo_21_cons_buff_0"} : memref<1x1x1x300xf32> 
    %fifo_21_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 5296 : i32, sym_name = "fifo_21_cons_buff_1"} : memref<1x1x1x300xf32> 
    %fifo_21_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 4) {init = 2 : i32, sym_name = "fifo_21_cons_prod_lock_0"}
    %fifo_21_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 5) {init = 0 : i32, sym_name = "fifo_21_cons_cons_lock_0"}
    %fifo_21_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 1 : i32, sym_name = "fifo_21_prod_lock_0"}
    %fifo_21_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "fifo_21_cons_lock_0"}
    %fifo_20_cons_buff_0 = aie.buffer(%tile_1_2) {address = 5120 : i32, sym_name = "fifo_20_cons_buff_0"} : memref<300xf32> 
    %fifo_20_cons_buff_1 = aie.buffer(%tile_1_2) {address = 6320 : i32, sym_name = "fifo_20_cons_buff_1"} : memref<300xf32> 
    %fifo_20_cons_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "fifo_20_cons_prod_lock_0"}
    %fifo_20_cons_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "fifo_20_cons_cons_lock_0"}
    %fifo_19_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 6496 : i32, sym_name = "fifo_19_cons_buff_0"} : memref<1x1x1x224xf32> 
    %fifo_19_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 7392 : i32, sym_name = "fifo_19_cons_buff_1"} : memref<1x1x1x224xf32> 
    %fifo_19_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 2) {init = 2 : i32, sym_name = "fifo_19_cons_prod_lock_0"}
    %fifo_19_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "fifo_19_cons_cons_lock_0"}
    %fifo_19_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 4) {init = 1 : i32, sym_name = "fifo_19_prod_lock_0"}
    %fifo_19_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 5) {init = 0 : i32, sym_name = "fifo_19_cons_lock_0"}
    %fifo_18_cons_buff_0 = aie.buffer(%tile_1_2) {address = 7520 : i32, sym_name = "fifo_18_cons_buff_0"} : memref<224xf32> 
    %fifo_18_cons_buff_1 = aie.buffer(%tile_1_2) {address = 8416 : i32, sym_name = "fifo_18_cons_buff_1"} : memref<224xf32> 
    %fifo_18_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "fifo_18_cons_prod_lock_0"}
    %fifo_18_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "fifo_18_cons_cons_lock_0"}
    %fifo_17_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 6) {init = 1 : i32, sym_name = "fifo_17_cons_prod_lock_0"}
    %fifo_17_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 7) {init = 0 : i32, sym_name = "fifo_17_cons_cons_lock_0"}
    %fifo_16_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "fifo_16_cons_buff_0"} : memref<512xf32> 
    %fifo_16_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 2048 : i32, sym_name = "fifo_16_cons_buff_1"} : memref<512xf32> 
    %fifo_16_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 10) {init = 2 : i32, sym_name = "fifo_16_cons_prod_lock_0"}
    %fifo_16_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 11) {init = 0 : i32, sym_name = "fifo_16_cons_cons_lock_0"}
    %fifo_16_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, sym_name = "fifo_16_buff_0"} : memref<512xf32> 
    %fifo_16_buff_1 = aie.buffer(%tile_0_4) {address = 3072 : i32, sym_name = "fifo_16_buff_1"} : memref<512xf32> 
    %fifo_16_prod_lock_0 = aie.lock(%tile_0_4, 4) {init = 2 : i32, sym_name = "fifo_16_prod_lock_0"}
    %fifo_16_cons_lock_0 = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "fifo_16_cons_lock_0"}
    %fifo_15_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 12992 : i32, sym_name = "fifo_15_cons_buff_0"} : memref<1x1x1x224xf32> 
    %fifo_15_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 13888 : i32, sym_name = "fifo_15_cons_buff_1"} : memref<1x1x1x224xf32> 
    %fifo_15_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 8) {init = 2 : i32, sym_name = "fifo_15_cons_prod_lock_0"}
    %fifo_15_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 9) {init = 0 : i32, sym_name = "fifo_15_cons_cons_lock_0"}
    %fifo_15_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 1 : i32, sym_name = "fifo_15_prod_lock_0"}
    %fifo_15_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "fifo_15_cons_lock_0"}
    %fifo_14_cons_buff_0 = aie.buffer(%tile_0_4) {address = 7520 : i32, sym_name = "fifo_14_cons_buff_0"} : memref<224xf32> 
    %fifo_14_cons_buff_1 = aie.buffer(%tile_0_4) {address = 8416 : i32, sym_name = "fifo_14_cons_buff_1"} : memref<224xf32> 
    %fifo_14_cons_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "fifo_14_cons_prod_lock_0"}
    %fifo_14_cons_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "fifo_14_cons_cons_lock_0"}
    %fifo_13_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 8192 : i32, sym_name = "fifo_13_cons_buff_0"} : memref<1x1x1x300xf32> 
    %fifo_13_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 9392 : i32, sym_name = "fifo_13_cons_buff_1"} : memref<1x1x1x300xf32> 
    %fifo_13_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 6) {init = 2 : i32, sym_name = "fifo_13_cons_prod_lock_0"}
    %fifo_13_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32, sym_name = "fifo_13_cons_cons_lock_0"}
    %fifo_13_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 1 : i32, sym_name = "fifo_13_prod_lock_0"}
    %fifo_13_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "fifo_13_cons_lock_0"}
    %fifo_12_cons_buff_0 = aie.buffer(%tile_0_4) {address = 5120 : i32, sym_name = "fifo_12_cons_buff_0"} : memref<300xf32> 
    %fifo_12_cons_buff_1 = aie.buffer(%tile_0_4) {address = 6320 : i32, sym_name = "fifo_12_cons_buff_1"} : memref<300xf32> 
    %fifo_12_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "fifo_12_cons_prod_lock_0"}
    %fifo_12_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "fifo_12_cons_cons_lock_0"}
    %fifo_11_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 1 : i32, sym_name = "fifo_11_cons_prod_lock_0"}
    %fifo_11_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "fifo_11_cons_cons_lock_0"}
    %fifo_10_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 4096 : i32, sym_name = "fifo_10_cons_buff_0"} : memref<512xf32> 
    %fifo_10_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 6144 : i32, sym_name = "fifo_10_cons_buff_1"} : memref<512xf32> 
    %fifo_10_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "fifo_10_cons_prod_lock_0"}
    %fifo_10_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "fifo_10_cons_cons_lock_0"}
    %fifo_10_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, sym_name = "fifo_10_buff_0"} : memref<512xf32> 
    %fifo_10_buff_1 = aie.buffer(%tile_0_3) {address = 3072 : i32, sym_name = "fifo_10_buff_1"} : memref<512xf32> 
    %fifo_10_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 2 : i32, sym_name = "fifo_10_prod_lock_0"}
    %fifo_10_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "fifo_10_cons_lock_0"}
    %fifo_9_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 14784 : i32, sym_name = "fifo_9_cons_buff_0"} : memref<1x1x1x224xf32> 
    %fifo_9_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 15680 : i32, sym_name = "fifo_9_cons_buff_1"} : memref<1x1x1x224xf32> 
    %fifo_9_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "fifo_9_cons_prod_lock_0"}
    %fifo_9_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "fifo_9_cons_cons_lock_0"}
    %fifo_9_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 1 : i32, sym_name = "fifo_9_prod_lock_0"}
    %fifo_9_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "fifo_9_cons_lock_0"}
    %fifo_8_cons_buff_0 = aie.buffer(%tile_0_3) {address = 7520 : i32, sym_name = "fifo_8_cons_buff_0"} : memref<224xf32> 
    %fifo_8_cons_buff_1 = aie.buffer(%tile_0_3) {address = 8416 : i32, sym_name = "fifo_8_cons_buff_1"} : memref<224xf32> 
    %fifo_8_cons_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "fifo_8_cons_prod_lock_0"}
    %fifo_8_cons_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "fifo_8_cons_cons_lock_0"}
    %fifo_7_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, sym_name = "fifo_7_cons_buff_0"} : memref<1x1x1x300xf32> 
    %fifo_7_cons_buff_1 = aie.buffer(%mem_tile_3_1) {address = 1200 : i32, sym_name = "fifo_7_cons_buff_1"} : memref<1x1x1x300xf32> 
    %fifo_7_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 0) {init = 2 : i32, sym_name = "fifo_7_cons_prod_lock_0"}
    %fifo_7_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 1) {init = 0 : i32, sym_name = "fifo_7_cons_cons_lock_0"}
    %fifo_7_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 1 : i32, sym_name = "fifo_7_prod_lock_0"}
    %fifo_7_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "fifo_7_cons_lock_0"}
    %fifo_6_cons_buff_0 = aie.buffer(%tile_0_3) {address = 5120 : i32, sym_name = "fifo_6_cons_buff_0"} : memref<300xf32> 
    %fifo_6_cons_buff_1 = aie.buffer(%tile_0_3) {address = 6320 : i32, sym_name = "fifo_6_cons_buff_1"} : memref<300xf32> 
    %fifo_6_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "fifo_6_cons_prod_lock_0"}
    %fifo_6_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "fifo_6_cons_cons_lock_0"}
    %fifo_5_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 1 : i32, sym_name = "fifo_5_cons_prod_lock_0"}
    %fifo_5_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "fifo_5_cons_cons_lock_0"}
    %fifo_4_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, sym_name = "fifo_4_cons_buff_0"} : memref<512xf32> 
    %fifo_4_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 2048 : i32, sym_name = "fifo_4_cons_buff_1"} : memref<512xf32> 
    %fifo_4_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "fifo_4_cons_prod_lock_0"}
    %fifo_4_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "fifo_4_cons_cons_lock_0"}
    %fifo_4_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "fifo_4_buff_0"} : memref<512xf32> 
    %fifo_4_buff_1 = aie.buffer(%tile_0_2) {address = 3072 : i32, sym_name = "fifo_4_buff_1"} : memref<512xf32> 
    %fifo_4_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "fifo_4_prod_lock_0"}
    %fifo_4_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "fifo_4_cons_lock_0"}
    %fifo_3_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 8288 : i32, sym_name = "fifo_3_cons_buff_0"} : memref<1x1x1x224xf32> 
    %fifo_3_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 9184 : i32, sym_name = "fifo_3_cons_buff_1"} : memref<1x1x1x224xf32> 
    %fifo_3_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "fifo_3_cons_prod_lock_0"}
    %fifo_3_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "fifo_3_cons_cons_lock_0"}
    %fifo_3_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 1 : i32, sym_name = "fifo_3_prod_lock_0"}
    %fifo_3_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "fifo_3_cons_lock_0"}
    %fifo_2_cons_buff_0 = aie.buffer(%tile_0_2) {address = 7520 : i32, sym_name = "fifo_2_cons_buff_0"} : memref<224xf32> 
    %fifo_2_cons_buff_1 = aie.buffer(%tile_0_2) {address = 8416 : i32, sym_name = "fifo_2_cons_buff_1"} : memref<224xf32> 
    %fifo_2_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "fifo_2_cons_prod_lock_0"}
    %fifo_2_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "fifo_2_cons_cons_lock_0"}
    %fifo_1_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 10592 : i32, sym_name = "fifo_1_cons_buff_0"} : memref<1x1x1x300xf32> 
    %fifo_1_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 11792 : i32, sym_name = "fifo_1_cons_buff_1"} : memref<1x1x1x300xf32> 
    %fifo_1_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "fifo_1_cons_prod_lock_0"}
    %fifo_1_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "fifo_1_cons_cons_lock_0"}
    %fifo_1_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 1 : i32, sym_name = "fifo_1_prod_lock_0"}
    %fifo_1_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "fifo_1_cons_lock_0"}
    %fifo_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 5120 : i32, sym_name = "fifo_0_cons_buff_0"} : memref<300xf32> 
    %fifo_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 6320 : i32, sym_name = "fifo_0_cons_buff_1"} : memref<300xf32> 
    %fifo_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "fifo_0_cons_prod_lock_0"}
    %fifo_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "fifo_0_cons_cons_lock_0"}
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<DMA : 0, North : 1>
      aie.connect<South : 3, DMA : 0>
      aie.connect<DMA : 1, North : 5>
      aie.connect<South : 5, DMA : 1>
      aie.connect<North : 1, DMA : 2>
      aie.connect<DMA : 2, South : 2>
      aie.connect<DMA : 3, North : 0>
      aie.connect<South : 2, DMA : 3>
      aie.connect<DMA : 4, North : 3>
      aie.connect<South : 4, DMA : 4>
      aie.connect<North : 3, DMA : 5>
      aie.connect<DMA : 5, South : 3>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<East : 3, DMA : 1>
      aie.connect<DMA : 0, East : 0>
      aie.connect<East : 2, North : 3>
      aie.connect<South : 5, North : 2>
      aie.connect<North : 1, South : 1>
      aie.connect<South : 0, North : 5>
      aie.connect<South : 3, North : 4>
      aie.connect<North : 0, South : 3>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 3>
      aie.connect<South : 7, North : 5>
      aie.connect<North : 2, South : 2>
      aie.connect<East : 2, North : 2>
      aie.connect<East : 0, North : 4>
      aie.connect<North : 3, South : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<DMA : 0, North : 1>
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 1, North : 5>
      aie.connect<South : 2, DMA : 1>
      aie.connect<DMA : 2, North : 0>
      aie.connect<South : 5, DMA : 2>
      aie.connect<North : 1, DMA : 3>
      aie.connect<DMA : 3, South : 2>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<South : 1, West : 3>
      aie.connect<West : 0, East : 1>
      aie.connect<East : 2, West : 2>
      aie.connect<South : 5, DMA : 0>
      aie.connect<South : 0, DMA : 1>
      aie.connect<DMA : 0, South : 1>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, West : 2>
      aie.connect<East : 2, West : 0>
      aie.connect<East : 1, North : 2>
      aie.connect<East : 0, North : 5>
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_2_1 = aie.switchbox(%mem_tile_2_1) {
      aie.connect<North : 2, DMA : 0>
      aie.connect<DMA : 0, South : 2>
    }
    %tile_2_2 = aie.tile(2, 2)
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<West : 1, South : 2>
      aie.connect<East : 1, West : 2>
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      aie.connect<North : 2, South : 2>
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 3, West : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_2_0 = aie.shim_mux(%shim_noc_tile_2_0) {
      aie.connect<North : 2, DMA : 0>
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<South : 2, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<South : 5, North : 2>
      aie.connect<South : 4, North : 0>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_3_1 = aie.switchbox(%mem_tile_3_1) {
      aie.connect<DMA : 0, North : 1>
      aie.connect<South : 0, DMA : 0>
    }
    %tile_3_2 = aie.tile(3, 2)
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<South : 1, West : 1>
    }
    %switchbox_3_0 = aie.switchbox(%shim_noc_tile_3_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, West : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_3_0 = aie.shim_mux(%shim_noc_tile_3_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<South : 2, DMA : 0>
      aie.connect<South : 0, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_4_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @conv2d_3x64_b1a_fp32(%fifo_0_cons_buff_0, %fifo_4_buff_0, %fifo_2_cons_buff_0) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
      aie.use_lock(%fifo_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_4_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @conv2d_3x64_b1a_fp32(%fifo_0_cons_buff_1, %fifo_4_buff_1, %fifo_2_cons_buff_1) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
      aie.use_lock(%fifo_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_4_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @conv2d_3x64_b1a_fp32(%fifo_0_cons_buff_0, %fifo_4_buff_0, %fifo_2_cons_buff_0) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
      aie.use_lock(%fifo_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.end
    } {link_with = "external0.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_10_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_8_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @conv2d_3x64_b1a_fp32(%fifo_6_cons_buff_0, %fifo_10_buff_0, %fifo_8_cons_buff_0) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
      aie.use_lock(%fifo_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_10_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_8_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_10_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_8_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @conv2d_3x64_b1a_fp32(%fifo_6_cons_buff_1, %fifo_10_buff_1, %fifo_8_cons_buff_1) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
      aie.use_lock(%fifo_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_10_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_8_cons_prod_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_10_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_8_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @conv2d_3x64_b1a_fp32(%fifo_6_cons_buff_0, %fifo_10_buff_0, %fifo_8_cons_buff_0) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
      aie.use_lock(%fifo_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_10_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_8_cons_prod_lock_0, Release, 1)
      aie.end
    } {link_with = "external0.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_12_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_16_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_14_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @conv2d_3x64_b1a_fp32(%fifo_12_cons_buff_0, %fifo_16_buff_0, %fifo_14_cons_buff_0) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
      aie.use_lock(%fifo_12_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_16_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_14_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_12_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_16_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_14_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @conv2d_3x64_b1a_fp32(%fifo_12_cons_buff_1, %fifo_16_buff_1, %fifo_14_cons_buff_1) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
      aie.use_lock(%fifo_12_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_16_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_14_cons_prod_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_12_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_16_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_14_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @conv2d_3x64_b1a_fp32(%fifo_12_cons_buff_0, %fifo_16_buff_0, %fifo_14_cons_buff_0) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
      aie.use_lock(%fifo_12_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_16_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_14_cons_prod_lock_0, Release, 1)
      aie.end
    } {link_with = "external0.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_20_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_22_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_18_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @conv2d_3x64_b1a_fp32(%fifo_20_cons_buff_0, %fifo_22_buff_0, %fifo_18_cons_buff_0) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
      aie.use_lock(%fifo_20_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_22_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_18_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_20_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_22_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_18_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @conv2d_3x64_b1a_fp32(%fifo_20_cons_buff_1, %fifo_22_buff_1, %fifo_18_cons_buff_1) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
      aie.use_lock(%fifo_20_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_22_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_18_cons_prod_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_20_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_22_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_18_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @conv2d_3x64_b1a_fp32(%fifo_20_cons_buff_0, %fifo_22_buff_0, %fifo_18_cons_buff_0) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
      aie.use_lock(%fifo_20_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_22_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_18_cons_prod_lock_0, Release, 1)
      aie.end
    } {link_with = "external0.o"}
    aiex.runtime_sequence(%arg0: memref<1048xf32>, %arg1: memref<1024xf32>, %arg2: memref<1048xf32>, %arg3: memref<1024xf32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 300][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<1048xf32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 600][1, 1, 1, 224][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<1048xf32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 512][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_5} : memref<1024xf32>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 1, 300][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_7} : memref<1048xf32>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 600][1, 1, 1, 224][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_9} : memref<1048xf32>
      aiex.npu.dma_memcpy_nd(%arg3[0, 0, 0, 0][1, 1, 1, 512][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_11} : memref<1024xf32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 300][1, 1, 1, 300][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_13} : memref<1048xf32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 824][1, 1, 1, 224][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_15} : memref<1048xf32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 512][1, 1, 1, 512][0, 0, 0, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_17} : memref<1024xf32>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 824][1, 1, 1, 224][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_19} : memref<1048xf32>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 300][1, 1, 1, 300][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_21} : memref<1048xf32>
      aiex.npu.dma_memcpy_nd(%arg3[0, 0, 0, 512][1, 1, 1, 512][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_23} : memref<1024xf32>
      aiex.npu.dma_wait {symbol = @fifo_5}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_wait {symbol = @fifo_17}
      aiex.npu.dma_wait {symbol = @fifo_23}
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_0 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_1 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_0 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_1 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%fifo_9_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_9_cons_buff_0 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%fifo_9_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_9_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_9_cons_buff_1 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%fifo_9_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%fifo_9_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_9_cons_buff_0 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%fifo_9_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%fifo_9_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_9_cons_buff_1 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%fifo_9_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%fifo_10_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_10_cons_buff_0 : memref<512xf32>, 0, 512) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_10_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%fifo_10_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_10_cons_buff_1 : memref<512xf32>, 0, 512) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_10_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%fifo_10_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_10_cons_buff_0 : memref<512xf32>, 0, 512) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%fifo_10_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%fifo_10_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_10_cons_buff_1 : memref<512xf32>, 0, 512) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%fifo_10_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(MM2S, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%fifo_13_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_13_cons_buff_0 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%fifo_13_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%fifo_13_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_13_cons_buff_1 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%fifo_13_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 3, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%fifo_13_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_13_cons_buff_0 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%fifo_13_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%fifo_13_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_13_cons_buff_1 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%fifo_13_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 4, ^bb25, ^bb27)
    ^bb25:  // 2 preds: ^bb24, ^bb26
      aie.use_lock(%fifo_15_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_15_cons_buff_0 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%fifo_15_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%fifo_15_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_15_cons_buff_1 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%fifo_15_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb25
    ^bb27:  // pred: ^bb24
      %9 = aie.dma_start(S2MM, 4, ^bb28, ^bb30)
    ^bb28:  // 2 preds: ^bb27, ^bb29
      aie.use_lock(%fifo_15_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_15_cons_buff_0 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%fifo_15_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%fifo_15_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_15_cons_buff_1 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%fifo_15_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb28
    ^bb30:  // pred: ^bb27
      %10 = aie.dma_start(S2MM, 5, ^bb31, ^bb33)
    ^bb31:  // 2 preds: ^bb30, ^bb32
      aie.use_lock(%fifo_16_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_cons_buff_0 : memref<512xf32>, 0, 512) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%fifo_16_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%fifo_16_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_cons_buff_1 : memref<512xf32>, 0, 512) {bd_id = 33 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%fifo_16_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb31
    ^bb33:  // pred: ^bb30
      %11 = aie.dma_start(MM2S, 5, ^bb34, ^bb36)
    ^bb34:  // 2 preds: ^bb33, ^bb35
      aie.use_lock(%fifo_16_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_cons_buff_0 : memref<512xf32>, 0, 512) {bd_id = 34 : i32, next_bd_id = 35 : i32}
      aie.use_lock(%fifo_16_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb35
    ^bb35:  // pred: ^bb34
      aie.use_lock(%fifo_16_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_cons_buff_1 : memref<512xf32>, 0, 512) {bd_id = 35 : i32, next_bd_id = 34 : i32}
      aie.use_lock(%fifo_16_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb34
    ^bb36:  // pred: ^bb33
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_cons_buff_0 : memref<300xf32>, 0, 300) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_cons_buff_1 : memref<300xf32>, 0, 300) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_0 : memref<224xf32>, 0, 224) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_1 : memref<224xf32>, 0, 224) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%fifo_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_buff_0 : memref<512xf32>, 0, 512) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_buff_1 : memref<512xf32>, 0, 512) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @fifo_1(MM2S, 0, 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_3_cons_buff_0 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_3_cons_buff_1 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_3_cons_buff_0 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_3_cons_buff_1 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%fifo_19_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_19_cons_buff_0 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%fifo_19_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_19_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_19_cons_buff_1 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%fifo_19_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%fifo_19_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_19_cons_buff_0 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%fifo_19_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%fifo_19_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_19_cons_buff_1 : memref<1x1x1x224xf32>, 0, 224) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%fifo_19_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(MM2S, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%fifo_21_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_21_cons_buff_0 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_21_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%fifo_21_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_21_cons_buff_1 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_21_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%fifo_21_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_21_cons_buff_0 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%fifo_21_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%fifo_21_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_21_cons_buff_1 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%fifo_21_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%fifo_22_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_22_cons_buff_0 : memref<512xf32>, 0, 512) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%fifo_22_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%fifo_22_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_22_cons_buff_1 : memref<512xf32>, 0, 512) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%fifo_22_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(MM2S, 3, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%fifo_22_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_22_cons_buff_0 : memref<512xf32>, 0, 512) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%fifo_22_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%fifo_22_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_22_cons_buff_1 : memref<512xf32>, 0, 512) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%fifo_22_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      aie.end
    }
    aie.shim_dma_allocation @fifo_3(MM2S, 0, 1)
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_0 : memref<512xf32>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_1 : memref<512xf32>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_0 : memref<512xf32>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_1 : memref<512xf32>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @fifo_5(S2MM, 0, 2)
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_7_cons_buff_0 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_7_cons_buff_1 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_7_cons_buff_0 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_7_cons_buff_1 : memref<1x1x1x300xf32>, 0, 300) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<300xf32>, 0, 300) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<300xf32>, 0, 300) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_8_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_8_cons_buff_0 : memref<224xf32>, 0, 224) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_8_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_8_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_8_cons_buff_1 : memref<224xf32>, 0, 224) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_8_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%fifo_10_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_10_buff_0 : memref<512xf32>, 0, 512) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_10_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_10_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_10_buff_1 : memref<512xf32>, 0, 512) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_10_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @fifo_7(MM2S, 0, 3)
    aie.shim_dma_allocation @fifo_9(MM2S, 1, 0)
    aie.shim_dma_allocation @fifo_11(S2MM, 0, 0)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_12_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_12_cons_buff_0 : memref<300xf32>, 0, 300) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_12_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_12_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_12_cons_buff_1 : memref<300xf32>, 0, 300) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_12_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_14_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_14_cons_buff_0 : memref<224xf32>, 0, 224) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_14_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_14_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_14_cons_buff_1 : memref<224xf32>, 0, 224) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_14_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%fifo_16_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_0 : memref<512xf32>, 0, 512) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_16_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_16_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_1 : memref<512xf32>, 0, 512) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_16_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @fifo_13(MM2S, 1, 1)
    aie.shim_dma_allocation @fifo_15(MM2S, 0, 2)
    aie.shim_dma_allocation @fifo_17(S2MM, 1, 0)
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_18_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_18_cons_buff_0 : memref<224xf32>, 0, 224) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_18_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_18_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_18_cons_buff_1 : memref<224xf32>, 0, 224) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_18_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_20_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_20_cons_buff_0 : memref<300xf32>, 0, 300) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_20_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_20_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_20_cons_buff_1 : memref<300xf32>, 0, 300) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_20_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%fifo_22_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_22_buff_0 : memref<512xf32>, 0, 512) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_22_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_22_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_22_buff_1 : memref<512xf32>, 0, 512) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_22_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @fifo_19(MM2S, 1, 2)
    aie.shim_dma_allocation @fifo_21(MM2S, 1, 3)
    aie.shim_dma_allocation @fifo_23(S2MM, 0, 1)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%tile_0_4 : Core, %switchbox_0_4 : Core)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
    aie.wire(%switchbox_1_0 : East, %switchbox_2_0 : West)
    aie.wire(%shim_mux_2_0 : North, %switchbox_2_0 : South)
    aie.wire(%shim_noc_tile_2_0 : DMA, %shim_mux_2_0 : DMA)
    aie.wire(%switchbox_1_1 : East, %switchbox_2_1 : West)
    aie.wire(%mem_tile_2_1 : Core, %switchbox_2_1 : Core)
    aie.wire(%mem_tile_2_1 : DMA, %switchbox_2_1 : DMA)
    aie.wire(%switchbox_2_0 : North, %switchbox_2_1 : South)
    aie.wire(%switchbox_1_2 : East, %switchbox_2_2 : West)
    aie.wire(%tile_2_2 : Core, %switchbox_2_2 : Core)
    aie.wire(%tile_2_2 : DMA, %switchbox_2_2 : DMA)
    aie.wire(%switchbox_2_1 : North, %switchbox_2_2 : South)
    aie.wire(%switchbox_2_0 : East, %switchbox_3_0 : West)
    aie.wire(%shim_mux_3_0 : North, %switchbox_3_0 : South)
    aie.wire(%shim_noc_tile_3_0 : DMA, %shim_mux_3_0 : DMA)
    aie.wire(%switchbox_2_1 : East, %switchbox_3_1 : West)
    aie.wire(%mem_tile_3_1 : Core, %switchbox_3_1 : Core)
    aie.wire(%mem_tile_3_1 : DMA, %switchbox_3_1 : DMA)
    aie.wire(%switchbox_3_0 : North, %switchbox_3_1 : South)
    aie.wire(%switchbox_2_2 : East, %switchbox_3_2 : West)
    aie.wire(%tile_3_2 : Core, %switchbox_3_2 : Core)
    aie.wire(%tile_3_2 : DMA, %switchbox_3_2 : DMA)
    aie.wire(%switchbox_3_1 : North, %switchbox_3_2 : South)
  }
}

