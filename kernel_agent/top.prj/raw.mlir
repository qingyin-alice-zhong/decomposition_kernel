module {
  func.func private @linear_12288x960_fp32(memref<32xf32>, memref<1xf32>, memref<32xf32>)
  func.func @core_0(%arg0: memref<32xf32>, %arg1: memref<1xf32>, %arg2: memref<32xf32>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "core_()"} {
    call @linear_12288x960_fp32(%arg0, %arg1, %arg2) : (memref<32xf32>, memref<1xf32>, memref<32xf32>) -> ()
    return
  }
  func.func @top(%arg0: memref<32xf32>, %arg1: memref<1xf32>, %arg2: memref<32xf32>) attributes {dataflow, itypes = "___"} {
    return
  }
}
