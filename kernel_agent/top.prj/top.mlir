module {
  aie.device(npu1_4col) {
    func.func private @conv2d_3x64_b1a_fp32(memref<300xf32>, memref<512xf32>, memref<224xf32>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_2 = aie.tile(1, 2)
    aie.objectfifo @fifo_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<300xf32>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x1x300xf32>> 
    aie.objectfifo @fifo_2(%mem_tile_1_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<224xf32>> 
    aie.objectfifo @fifo_3(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x1x224xf32>> 
    aie.objectfifo @fifo_4(%tile_0_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<512xf32>> 
    aie.objectfifo @fifo_5(%mem_tile_2_1, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<1x1x1x512xf32>> 
    aie.objectfifo @fifo_6(%mem_tile_3_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<300xf32>> 
    aie.objectfifo @fifo_7(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1x1x1x300xf32>> 
    aie.objectfifo @fifo_8(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<224xf32>> 
    aie.objectfifo @fifo_9(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x1x224xf32>> 
    aie.objectfifo @fifo_10(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<512xf32>> 
    aie.objectfifo @fifo_11(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x1x1x512xf32>> 
    aie.objectfifo @fifo_12(%mem_tile_0_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<300xf32>> 
    aie.objectfifo @fifo_13(%shim_noc_tile_1_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x1x300xf32>> 
    aie.objectfifo @fifo_14(%mem_tile_0_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<224xf32>> 
    aie.objectfifo @fifo_15(%shim_noc_tile_2_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x1x224xf32>> 
    aie.objectfifo @fifo_16(%tile_0_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<512xf32>> 
    aie.objectfifo @fifo_17(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x1x1x512xf32>> 
    aie.objectfifo @fifo_18(%mem_tile_1_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<224xf32>> 
    aie.objectfifo @fifo_19(%shim_noc_tile_2_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x1x224xf32>> 
    aie.objectfifo @fifo_20(%mem_tile_1_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<300xf32>> 
    aie.objectfifo @fifo_21(%shim_noc_tile_3_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x1x300xf32>> 
    aie.objectfifo @fifo_22(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<512xf32>> 
    aie.objectfifo @fifo_23(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x1x1x512xf32>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_9] -> [@fifo_8]([] [])
    aie.objectfifo.link [@fifo_10] -> [@fifo_11]([] [])
    aie.objectfifo.link [@fifo_13] -> [@fifo_12]([] [])
    aie.objectfifo.link [@fifo_15] -> [@fifo_14]([] [])
    aie.objectfifo.link [@fifo_16] -> [@fifo_17]([] [])
    aie.objectfifo.link [@fifo_3] -> [@fifo_2]([] [])
    aie.objectfifo.link [@fifo_19] -> [@fifo_18]([] [])
    aie.objectfifo.link [@fifo_21] -> [@fifo_20]([] [])
    aie.objectfifo.link [@fifo_22] -> [@fifo_23]([] [])
    aie.objectfifo.link [@fifo_4] -> [@fifo_5]([] [])
    aie.objectfifo.link [@fifo_7] -> [@fifo_6]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<300xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<300xf32>> -> memref<300xf32>
        %2 = aie.objectfifo.acquire @fifo_4(Produce, 1) : !aie.objectfifosubview<memref<512xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<512xf32>> -> memref<512xf32>
        %4 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<224xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<224xf32>> -> memref<224xf32>
        func.call @conv2d_3x64_b1a_fp32(%1, %3, %5) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_4(Produce, 1)
        aie.objectfifo.release @fifo_2(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<300xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<300xf32>> -> memref<300xf32>
        %2 = aie.objectfifo.acquire @fifo_10(Produce, 1) : !aie.objectfifosubview<memref<512xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<512xf32>> -> memref<512xf32>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<224xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<224xf32>> -> memref<224xf32>
        func.call @conv2d_3x64_b1a_fp32(%1, %3, %5) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_10(Produce, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<300xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<300xf32>> -> memref<300xf32>
        %2 = aie.objectfifo.acquire @fifo_16(Produce, 1) : !aie.objectfifosubview<memref<512xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<512xf32>> -> memref<512xf32>
        %4 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<224xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<224xf32>> -> memref<224xf32>
        func.call @conv2d_3x64_b1a_fp32(%1, %3, %5) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
        aie.objectfifo.release @fifo_12(Consume, 1)
        aie.objectfifo.release @fifo_16(Produce, 1)
        aie.objectfifo.release @fifo_14(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_20(Consume, 1) : !aie.objectfifosubview<memref<300xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<300xf32>> -> memref<300xf32>
        %2 = aie.objectfifo.acquire @fifo_22(Produce, 1) : !aie.objectfifosubview<memref<512xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<512xf32>> -> memref<512xf32>
        %4 = aie.objectfifo.acquire @fifo_18(Consume, 1) : !aie.objectfifosubview<memref<224xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<224xf32>> -> memref<224xf32>
        func.call @conv2d_3x64_b1a_fp32(%1, %3, %5) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
        aie.objectfifo.release @fifo_20(Consume, 1)
        aie.objectfifo.release @fifo_22(Produce, 1)
        aie.objectfifo.release @fifo_18(Consume, 1)
      }
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
  }
}
