; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@fifo_0_cons_buff_1 = external global [972 x float]
@fifo_0_cons_buff_0 = external global [972 x float]
@fifo_1_cons_buff_1 = external global [1 x [1 x [1 x [972 x float]]]]
@fifo_1_cons_buff_0 = external global [1 x [1 x [1 x [972 x float]]]]
@fifo_2_cons_buff_1 = external global [224 x float]
@fifo_2_cons_buff_0 = external global [224 x float]
@fifo_3_cons_buff_1 = external global [1 x [1 x [1 x [224 x float]]]]
@fifo_3_cons_buff_0 = external global [1 x [1 x [1 x [224 x float]]]]
@fifo_4_buff_1 = external global [2048 x float]
@fifo_4_buff_0 = external global [2048 x float]
@fifo_4_cons_buff_1 = external global [2048 x float]
@fifo_4_cons_buff_0 = external global [2048 x float]
@fifo_5_cons = external global [1 x [1 x [1 x [2048 x float]]]]
@fifo_5 = external global [1 x [1 x [1 x [2048 x float]]]]
@fifo_4_cons = external global [2048 x float]
@fifo_4 = external global [2048 x float]
@fifo_3_cons = external global [1 x [1 x [1 x [224 x float]]]]
@fifo_3 = external global [1 x [1 x [1 x [224 x float]]]]
@fifo_2_cons = external global [224 x float]
@fifo_2 = external global [224 x float]
@fifo_1_cons = external global [1 x [1 x [1 x [972 x float]]]]
@fifo_1 = external global [1 x [1 x [1 x [972 x float]]]]
@fifo_0_cons = external global [972 x float]
@fifo_0 = external global [972 x float]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

declare void @conv2d_3x64_b1a_fp32(ptr, ptr, ptr)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %5, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775806
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_4_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_0, i64 32) ]
  call void @conv2d_3x64_b1a_fp32(ptr @fifo_0_cons_buff_0, ptr @fifo_4_buff_0, ptr @fifo_2_cons_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_4_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_1, i64 32) ]
  call void @conv2d_3x64_b1a_fp32(ptr @fifo_0_cons_buff_1, ptr @fifo_4_buff_1, ptr @fifo_2_cons_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %5 = add i64 %2, 2
  br label %1

6:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_4_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_0, i64 32) ]
  call void @conv2d_3x64_b1a_fp32(ptr @fifo_0_cons_buff_0, ptr @fifo_4_buff_0, ptr @fifo_2_cons_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
