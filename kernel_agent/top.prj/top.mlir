module {
  aie.device(npu1_4col) {
    func.func private @conv2d_3x64_b1a_fp32(memref<972xf32>, memref<2048xf32>, memref<224xf32>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @fifo_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<972xf32>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x1x972xf32>> 
    aie.objectfifo @fifo_2(%mem_tile_1_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<224xf32>> 
    aie.objectfifo @fifo_3(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x1x224xf32>> 
    aie.objectfifo @fifo_4(%tile_0_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<2048xf32>> 
    aie.objectfifo @fifo_5(%mem_tile_2_1, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<1x1x1x2048xf32>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_3] -> [@fifo_2]([] [])
    aie.objectfifo.link [@fifo_4] -> [@fifo_5]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<972xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<972xf32>> -> memref<972xf32>
        %2 = aie.objectfifo.acquire @fifo_4(Produce, 1) : !aie.objectfifosubview<memref<2048xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2048xf32>> -> memref<2048xf32>
        %4 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<224xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<224xf32>> -> memref<224xf32>
        func.call @conv2d_3x64_b1a_fp32(%1, %3, %5) : (memref<972xf32>, memref<2048xf32>, memref<224xf32>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_4(Produce, 1)
        aie.objectfifo.release @fifo_2(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_2_0, DMA : 1>
    } {keep_pkt_header = true}
    aiex.runtime_sequence(%arg0: memref<972xf32>, %arg1: memref<2048xf32>, %arg2: memref<224xf32>) {
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 2 : i32, value = 1 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 2 : i32, value = 1260724769 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 2 : i32, value = 439168079 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 2 : i32, value = 289 : ui32}
      aiex.npu.write32 {address = 261892 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 262144 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 2 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 1 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67228132 : ui32, arg_idx = 3 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119308 : ui32, column = 2 : i32, row = 0 : i32, value = 15 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 2 : i32, row = 0 : i32, value = 32512 : ui32}
      aiex.npu.write32 {address = 213068 : ui32, column = 2 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 2 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 972][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<972xf32>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 1, 224][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<224xf32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_5} : memref<2048xf32>
      aiex.npu.dma_wait {symbol = @fifo_5}
      aie.end
    }
  }
}
