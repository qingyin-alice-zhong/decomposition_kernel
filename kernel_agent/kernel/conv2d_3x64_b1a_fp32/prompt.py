PROMPT = """
你现在要作为一个自动调试与实现代理，专门负责更新 decomposition_workspace/kernel_agent/kernel/conv2d_3x64_b1a_fp32/conv2d_3x64_b1a_fp32.cc，使其尽可能正确实现 decomposition_workspace/kernel_agent/kernel/conv2d_3x64_b1a_fp32/conv2d_3x64_b1a_fp32.py 所表达的功能语义。

任务要求如下：

1. 目标文件
- 目标 C++ 文件是 decomposition_workspace/kernel_agent/kernel/conv2d_3x64_b1a_fp32/conv2d_3x64_b1a_fp32.cc。
- 如果这个 .cc 文件不存在，就直接创建它。
- 如果它已经存在，就在现有基础上继续修改，而不是无关重写，除非你确认现有实现方向明显错误。

2. 功能目标
- 你的目标是让它尽可能实现 decomposition_workspace/kernel_agent/kernel/conv2d_3x64_b1a_fp32/conv2d_3x64_b1a_fp32.py 中所描述的 Conv2d(3, 64, kernel_size=3, padding=1) 的功能语义。


3. 参考依据
- 你可以参考 backends/npu_new/api_doc.md 中相关内容。
- 你可以参考 decomposition_workspace/kernel_agent/kernel/conv2d_3x64_b1a_fp32/conv2d_3x64_b1a_fp32_test.py 的测试组织方式、tile 定义、参数布局、reference 逻辑、输出验证方式。


4. 工作方式
- 每次只做一轮有根据的更新，不要一次性做大量不可解释的改动。
- 每次更新 .cc 文件后，必须暂停这次 agent 活动，不要继续自行假设结果。
- 暂停时，明确提示用户去终端手动运行下面这条命令：
	python ~/decomposition_workspace/kernel_agent/main.py --kernel conv2d_3x64_b1a_fp32
- 然后等待用户把终端运行结果回复给你。

5. 迭代调试规则
- 如果用户反馈运行失败、verification 不通过、编译报错、执行异常、出现 NaN、buffer 超限等问题，你可以查看 decomposition_workspace/kernel_agent/outputs 中最新一次运行输出。
- 你应当根据最新一次 outputs 中的 compile_output、summary.json、execution_output、trace 信息，分析失败原因。
- 然后继续对 .cc 文件做下一轮有针对性的修改。
- 每一轮修改后，仍然必须再次暂停，并提示用户手动运行：
	python ~/decomposition_workspace/kernel_agent/main.py --kernel conv2d_3x64_b1a_fp32
- 继续等待用户反馈，形成“修改 -> 用户运行 -> 根据最新输出继续 debug -> 再修改”的循环。

6. 修改原则
- 优先做最小必要修改。
- 每次修改时要说明你在修什么问题，例如：参数解释错误、float32 路径不稳定、局部缓冲问题、tile 访存越界、padding 处理错误、输出布局错误、NaN 来源、AIE API 使用方式不当等。
- 不要把问题模糊地归因于“可能环境有问题”，除非日志里有直接证据。
- 如果发现当前 test.py 的组织方式已经限定了 kernel 的输入输出形状或 tile 语义，就要让 .cc 严格服从 test.py，而不是反过来要求用户先改 test.py。

7. 输出要求
- 每一轮修改后，先简要说明本轮你对 .cc 做了什么修改、为什么这么改。
- 然后立刻停止继续操作，并明确提示用户运行：
	python ~/decomposition_workspace/kernel_agent/main.py --kernel conv2d_3x64_b1a_fp32
- 不要直接替用户执行该命令。
- 等用户返回运行结果后，再继续下一轮分析和更新。

8. 总目标
- 通过多轮迭代，让 decomposition_workspace/kernel_agent/kernel/conv2d_3x64_b1a_fp32/conv2d_3x64_b1a_fp32.cc 尽可能达到：
	1) 能编译；
	2) 能执行；
	3) verification 尽可能通过；
	4) 若 verification 暂时无法通过，也要尽可能把错误范围缩小并定位到具体原因。


"""
