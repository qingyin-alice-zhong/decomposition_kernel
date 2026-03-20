module {
  func.func private @conv2d_3x64_b1a_fp32(memref<972xf32>, memref<2048xf32>, memref<224xf32>)
  func.func @core_0(%arg0: memref<972xf32>, %arg1: memref<2048xf32>, %arg2: memref<224xf32>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "core_()"} {
    call @conv2d_3x64_b1a_fp32(%arg0, %arg1, %arg2) : (memref<972xf32>, memref<2048xf32>, memref<224xf32>) -> ()
    return
  }
  func.func @top(%arg0: memref<972xf32>, %arg1: memref<2048xf32>, %arg2: memref<224xf32>) attributes {dataflow, itypes = "___"} {
    return
  }
}
