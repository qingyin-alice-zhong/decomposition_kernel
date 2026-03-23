module {
  func.func private @conv2d_3x64_b1a_fp32(memref<300xf32>, memref<512xf32>, memref<224xf32>)
  func.func @core_0(%arg0: memref<300xf32>, %arg1: memref<300xf32>, %arg2: memref<300xf32>, %arg3: memref<300xf32>, %arg4: memref<512xf32>, %arg5: memref<512xf32>, %arg6: memref<512xf32>, %arg7: memref<512xf32>, %arg8: memref<224xf32>, %arg9: memref<224xf32>, %arg10: memref<224xf32>, %arg11: memref<224xf32>) attributes {df.kernel, itypes = "____________", otypes = "", stypes = "____________", tag = "core_((), None, None, ())"} {
    call @conv2d_3x64_b1a_fp32(%arg0, %arg4, %arg8) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
    return
  }
  func.func @core_1(%arg0: memref<300xf32>, %arg1: memref<300xf32>, %arg2: memref<300xf32>, %arg3: memref<300xf32>, %arg4: memref<512xf32>, %arg5: memref<512xf32>, %arg6: memref<512xf32>, %arg7: memref<512xf32>, %arg8: memref<224xf32>, %arg9: memref<224xf32>, %arg10: memref<224xf32>, %arg11: memref<224xf32>) attributes {df.kernel, itypes = "____________", otypes = "", stypes = "____________", tag = "core_(None, (), None, ())"} {
    call @conv2d_3x64_b1a_fp32(%arg1, %arg5, %arg9) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
    return
  }
  func.func @core_2(%arg0: memref<300xf32>, %arg1: memref<300xf32>, %arg2: memref<300xf32>, %arg3: memref<300xf32>, %arg4: memref<512xf32>, %arg5: memref<512xf32>, %arg6: memref<512xf32>, %arg7: memref<512xf32>, %arg8: memref<224xf32>, %arg9: memref<224xf32>, %arg10: memref<224xf32>, %arg11: memref<224xf32>) attributes {df.kernel, itypes = "____________", otypes = "", stypes = "____________", tag = "core_(None, None, (), None)"} {
    call @conv2d_3x64_b1a_fp32(%arg2, %arg6, %arg10) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
    return
  }
  func.func @core_3(%arg0: memref<300xf32>, %arg1: memref<300xf32>, %arg2: memref<300xf32>, %arg3: memref<300xf32>, %arg4: memref<512xf32>, %arg5: memref<512xf32>, %arg6: memref<512xf32>, %arg7: memref<512xf32>, %arg8: memref<224xf32>, %arg9: memref<224xf32>, %arg10: memref<224xf32>, %arg11: memref<224xf32>) attributes {df.kernel, itypes = "____________", otypes = "", stypes = "____________", tag = "core_(None, None, None, ())"} {
    call @conv2d_3x64_b1a_fp32(%arg3, %arg7, %arg11) : (memref<300xf32>, memref<512xf32>, memref<224xf32>) -> ()
    return
  }
  func.func @top(%arg0: memref<300xf32>, %arg1: memref<300xf32>, %arg2: memref<300xf32>, %arg3: memref<300xf32>, %arg4: memref<512xf32>, %arg5: memref<512xf32>, %arg6: memref<512xf32>, %arg7: memref<512xf32>, %arg8: memref<224xf32>, %arg9: memref<224xf32>, %arg10: memref<224xf32>, %arg11: memref<224xf32>) attributes {dataflow, itypes = "____________"} {
    return
  }
}
