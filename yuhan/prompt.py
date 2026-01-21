"""
EDA Schema Extraction Prompts
用于从EDA工具文档中提取结构化信息

使用方法:
1. Stage 1: 使用 STAGE1_IDENTIFY_CLASSES 识别文档中的schema类型及实例
2. Stage 2: 根据识别结果，使用对应类型的prompt提取详细字段
3. Stage 3: 使用 verify_schemas() 验证关联关系并生成报告
"""

import re

# =============================================================================
# Stage 1: 识别Schema类型和实例
# =============================================================================

STAGE1_IDENTIFY_CLASSES = """你是EDA工具文档分析专家。请分析以下文档，识别其中包含的信息类型及具体实例。

【Schema类型说明】

1. **Command (指令)**
   工具可执行的命令，通常为下划线连接的英文单词，如 fix_setup_path_violations, read_timing_data, write_design_changes。
   文档中会描述其语法、参数、选项等。一个文档可能包含多个Command。

2. **Argument (选项)**
   命令的可选开关或标志，通常以 - 或 -- 开头，如 -size_down_only, -reorder, -force。
   用于修改命令的默认行为。一个Command可能有多个Argument。

3. **Parameter (参数)**
   工具的变量，如 eco_post_mask_mode, max_thread_number, eco_ga_auto_refill。

4. **Example (示例)**
   文档中给出的具体命令用法示例，包含可直接执行的代码片段。
   如 "xtop > set_parameter eco_post_mask_mode true"

5. **Mode (模式)**
   工具的运行模式，如 Turbo Mode, Normal Mode, Pro Mode。
   一定是xxx mode的形式

6. **File (文件)**
   工具使用或生成的文件类型，如 timing data文件, verilog文件, sta log文件。

7. **FailReasons (失败原因)**
   工具的failreason，是文档定义的一类特殊概念
   只有文档明确该概念表示是fail reason才算

8. **Issues (问题)**
   针对某个问题或者目的的长文字描述，如FAQ类内容。
   可以简要总结出问题或者目的的一段话，一般包含复杂的逻辑

9. **Task (任务)**
   EDA流程中的具体任务目标，如 fix setup violations, fix hold, leakage优化, 加速。

10. **Concept (概念)**
    EDA领域的专业术语或概念，如 hold slack, setup timing, POCV, derate, 反标率, 5nm工艺。

11. **Operation (操作)**
    完成特定目标的操作，如"启用postmask ECO模式"、"设置dont touch"。

【文档内容】
{document}

【文档结构说明】（可选）
{document_notes}

【输出要求】
以JSON格式输出，列出所有识别到的类型及其实例名称：
{{
  "identified_classes": {{
    "Command": ["命令1", "命令2"],
    "Parameter": ["参数1", "参数2"],
    "Mode": ["模式1"],
    ...
  }}
}}

注意：只输出文档中确实存在的类型，没有的类型不要列出。

请输出JSON："""


# =============================================================================
# Stage 1: Glean (多轮补充提取)
# =============================================================================

STAGE1_GLEAN = """你是EDA文档专家。请检查是否有遗漏的实例。

【已识别】
{previous_results}

【文档】
{document}

【输出】遗漏的实例（无遗漏则列表为空）：
{{"additional_instances": {{"Command": [], "Argument": [], "Parameter": [], "Example": [], "Mode": [], "File": [], "FailReasons": [], "Issues": [], "Task": [], "Concept": [], "Operation": []}}}}

请输出JSON："""


# =============================================================================
# Stage 2: Command 提取
# =============================================================================

STAGE2_EXTRACT_COMMAND = """你是EDA工具文档专家。文档中有关于指令{instance}的内容，请阅读文档并从中提取所有与该命令相关的内容，填充到【输出格式】的字段中。

【输出格式】
{{
  "class": "Command",
  "name": "指令名称",
  "usage": "指令功能，描述这个命令做什么",
  "syntax": "完整语法格式，包含所有选项和参数的占位符",
  "arguments"（选项列表，每个元素为选项名字，不需要对选项进行解释）: [
    "-arg1",
    ...
  ],
  "values": （参数列表）[
    {{
      "usage": "第一个参数的含义",
      "type": "类型(string/int/...)",
      "optional": 是否可选（true/false/null）,
      "range": [min, max]（如果是离散则没有range字段）,
      "key_values"（关键取值）: [
        {{
          "value": "关键值1",
          "usage": "关键值1的含义",
          "scenarios": "关键值1的使用场景"
        }},
        ...（如果有多个关键词）
      ]
    }},
    ... （如果有多个参数）
  ],
  "scenarios": "命令的适用场景，什么时候使用这个命令"
}}

【文档内容】
{document}

【注意】
1. 对于`cmd val2 -arg1 val1`，val2是values，arg1是arguments（val1是arg1的参数，不用提取）。
2. 如果有字段内容文档没有明确提到，则该字段为null

请输出JSON："""


# =============================================================================
# Stage 2: Argument 提取
# =============================================================================

STAGE2_EXTRACT_ARGUMENT = """你是EDA工具文档专家。文档中有关于选项{instance}的内容，请阅读文档并从中提取所有与该选项相关的内容，填充到【输出格式】的字段中。

【输出格式】
{{
  "class": "Argument",
  "name": "选项名称",
  "command": "该选项所属的命令名称",
  "usage": "选项功能，描述启用这个选项后的效果",
  "syntax": "选项的语法格式，如 -arg1 <value>",
  "values"（选项的参数列表，如果选项不需要参数则为null）: [
    {{
      "usage": "第一个参数的含义",
      "type": "类型(string/int/...)",
      "optional": 是否可选（true/false/null）,
      "range": [min, max]（如果是连续取值，否则没有range字段）,
      "key_values"（关键取值）: [
        {{
          "value": "关键值1",
          "usage": "关键值1的含义",
          "scenarios": "关键值1的使用场景"
        }},
        ...（如果有多个关键值）
      ]
    }},
    ...（如果有多个参数）
  ],
  "scenarios": "选项的适用场景，什么情况下应该使用这个选项"
}}

【文档内容】
{document}

【注意】
1. command字段填写该选项所属的命令名称
2. 选项的参数指的是选项后面紧跟的值，如`-size_down_only true`中true是参数
3. 如果选项是开关型（无参数），values字段为null
4. 如果有字段内容文档没有明确提到，则该字段为null

请输出JSON："""


# =============================================================================
# Stage 2: Parameter 提取
# =============================================================================

STAGE2_EXTRACT_PARAMETER = """你是EDA工具文档专家。文档中有关于参数{instance}的内容，请阅读文档并从中提取所有与该参数相关的内容，填充到【输出格式】的字段中。

【输出格式】
{{
  "class": "Parameter",
  "name": "参数名称",
  "usage": "参数功能，描述这个参数控制什么行为",
  "type": "数据类型(boolean/string/int/float/list)",
  "range": [min, max]（如果是连续取值，否则没有range字段）,
  "key_values"（关键取值）: [
    {{
      "value": "关键值1",
      "usage": "关键值1的含义",
      "scenarios": "关键值1的使用场景"
    }},
    ...（如果有多个关键值）
  ],
  "scenarios": "参数的适用场景，什么情况下需要设置这个参数"
}}

【文档内容】
{document}

【注意】
1. 对于boolean类型，key_values应包含true和false两个值及其含义
2. 如果有字段内容文档没有明确提到，则该字段为null

请输出JSON："""


# =============================================================================
# Stage 2: Example 提取
# =============================================================================

STAGE2_EXTRACT_EXAMPLE = """你是EDA工具文档专家。文档中有关于示例{instance}的内容，请阅读文档并从中提取所有与该示例相关的内容，填充到【输出格式】的字段中。

【输出格式】
{{
  "class": "Example",
  "name": "示例代码或命令内容，与原文完全一样",
  "usage": "这个示例演示什么功能",
  "scenarios": "这个示例适用于什么场景"
}}

【文档内容】
{document}

【注意】
1. name字段应保留原文的完整示例代码
2. 如果有字段内容文档没有明确提到，则该字段为null

请输出JSON："""


# =============================================================================
# Stage 2: Mode 提取
# =============================================================================

STAGE2_EXTRACT_MODE = """你是EDA工具文档专家。文档中有关于模式{instance}的内容，请阅读文档并从中提取所有与该模式相关的内容，填充到【输出格式】的字段中。

【输出格式】
{{
  "class": "Mode",
  "name": "模式名称，如Turbo Mode",
  "usage": "模式特点，包括速度、内存、精度等方面的特性，以及与其他模式的对比",
  "scenarios": "适用场景，什么情况下应该选择这个模式"
}}

【文档内容】
{document}

【注意】
1. 如果文档中有与其他模式的对比信息，应包含在usage中
2. 如果有字段内容文档没有明确提到，则该字段为null

请输出JSON："""


# =============================================================================
# Stage 2: File 提取
# =============================================================================

STAGE2_EXTRACT_FILE = """你是EDA工具文档专家。文档中有关于文件{instance}的内容，请阅读文档并从中提取所有与该文件相关的内容，填充到【输出格式】的字段中。

【输出格式】
{{
  "class": "File",
  "name": "文件名称或类型",
  "usage": "文件用途，描述这个文件包含什么内容、如何使用"
}}

【文档内容】
{document}

【注意】
1. 如果有字段内容文档没有明确提到，则该字段为null

请输出JSON："""


# =============================================================================
# Stage 2: FailReasons 提取
# =============================================================================

STAGE2_EXTRACT_FAILREASONS = """你是EDA工具文档专家。文档中有关于失败原因{instance}的内容，请阅读文档并从中提取所有与该失败原因相关的内容，填充到【输出格式】的字段中。

【输出格式】
{{
  "class": "FailReasons",
  "name": "失败原因的简短名称",
  "reasons": ["导致失败的具体原因1", "原因2", ...],
  "description": "详细解释",
  "solution": "解决方案，包括具体的命令、参数设置步骤"
}}

【文档内容】
{document}

【注意】
1. reasons是一个列表，包含所有可能导致该失败的原因
2. solution应尽量具体，包含可执行的命令或参数设置
3. 如果有字段内容文档没有明确提到，则该字段为null

请输出JSON："""


# =============================================================================
# Stage 2: Issues 提取
# =============================================================================

STAGE2_EXTRACT_ISSUES = """你是EDA工具文档专家。文档中有关于问题{instance}的内容，请阅读文档并从中提取所有与该问题相关的内容，填充到【输出格式】的字段中。

【输出格式】
{{
  "class": "Issues",
  "name": "要解决的问题或者要实现的目的的简短描述",
  "descriptions": "详细解答、解释或者步骤等"
}}

【文档内容】
{document}

【注意】
1. descriptions应包含完整的解答逻辑和相关细节
2. 如果有字段内容文档没有明确提到，则该字段为null

请输出JSON："""


# =============================================================================
# Stage 2: Task 提取
# =============================================================================

STAGE2_EXTRACT_TASK = """你是EDA工具文档专家。文档中有关于任务{instance}的内容，请阅读文档并从中提取所有与该任务相关的内容，填充到【输出格式】的字段中。

【输出格式】
{{
  "class": "Task",
  "name": "EDA任务名称，如fix setup violations, hold优化",
  "description": "任务描述，包括任务目标、涉及的步骤、注意事项"
}}

【文档内容】
{document}

【注意】
1. description应包含任务的完整流程和关键步骤
2. 如果有字段内容文档没有明确提到，则该字段为null

请输出JSON："""


# =============================================================================
# Stage 2: Concept 提取
# =============================================================================

STAGE2_EXTRACT_CONCEPT = """你是EDA工具文档专家。文档中有关于概念{instance}的内容，请阅读文档并从中提取所有与该概念相关的内容，填充到【输出格式】的字段中。

【输出格式】
{{
  "class": "Concept",
  "name": "EDA概念名称，如hold slack, POCV, derate",
  "description": "概念解释，说明这个概念的含义、作用、影响因素"
}}

【文档内容】
{document}

【注意】
1. description应用通俗易懂的方式解释概念
2. 如果有字段内容文档没有明确提到，则该字段为null

请输出JSON："""


# =============================================================================
# Stage 2: Operation 提取
# =============================================================================

STAGE2_EXTRACT_OPERATION = """你是EDA工具文档专家。文档中有关于操作{instance}的内容，请阅读文档并从中提取所有与该操作相关的内容，填充到【输出格式】的字段中。

【输出格式】
{{
  "class": "Operation",
  "name": "EDA操作名称，如启用postmask模式、设置dont touch",
  "description": "操作步骤的详细描述",
  "related_commands": ["涉及的命令1", "涉及的参数2", ...],
  "effect": "执行该操作后的效果和影响"
}}

【文档内容】
{document}

【注意】
1. related_commands应列出所有相关的命令、参数、选项
2. effect应描述操作的预期结果和可能的副作用
3. 如果有字段内容文档没有明确提到，则该字段为null

请输出JSON："""


# =============================================================================
# Prompt映射表
# =============================================================================

STAGE2_PROMPTS = {
    "Command": STAGE2_EXTRACT_COMMAND,
    "Argument": STAGE2_EXTRACT_ARGUMENT,
    "Parameter": STAGE2_EXTRACT_PARAMETER,
    "Example": STAGE2_EXTRACT_EXAMPLE,
    "Mode": STAGE2_EXTRACT_MODE,
    "File": STAGE2_EXTRACT_FILE,
    "FailReasons": STAGE2_EXTRACT_FAILREASONS,
    "Issues": STAGE2_EXTRACT_ISSUES,
    "Task": STAGE2_EXTRACT_TASK,
    "Concept": STAGE2_EXTRACT_CONCEPT,
    "Operation": STAGE2_EXTRACT_OPERATION
}


# =============================================================================
# Stage 3: 验证逻辑
# =============================================================================

# 命名格式正则
COMMAND_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')
ARGUMENT_PATTERN = re.compile(r'^-?[a-z][a-z0-9_]*$')


def verify_schemas(commands, arguments, parameters, operations):
    """
    验证schema实例之间的关联关系，生成警告报告。

    Args:
        commands: Command实例列表，每个包含 {name, arguments: [...]}
        arguments: Argument实例列表，每个包含 {name, command}
        parameters: Parameter实例列表，每个包含 {name}
        operations: Operation实例列表，每个包含 {name, related_commands: [...]}

    Returns:
        str: 警告报告
    """
    report = ["=== Schema Verification Report ===\n"]

    # 收集所有名称
    cmd_names = {c.get('name') for c in commands if c.get('name')}
    arg_names = {a.get('name') for a in arguments if a.get('name')}
    param_names = {p.get('name') for p in parameters if p.get('name')}

    # 1. 格式验证
    invalid_format = []
    for c in commands:
        name = c.get('name', '')
        if name and not COMMAND_PATTERN.match(name):
            invalid_format.append(f"Command \"{name}\" does not match pattern [a-z][a-z0-9_]*")
    for a in arguments:
        name = a.get('name', '')
        if name and not ARGUMENT_PATTERN.match(name):
            invalid_format.append(f"Argument \"{name}\" does not match pattern -?[a-z][a-z0-9_]*")

    if invalid_format:
        report.append("[Invalid Format]")
        report.extend(f"- {msg}" for msg in invalid_format)
        report.append("")

    # 2. Command.arguments -> Argument实例
    missing_arg_instances = []
    for c in commands:
        cmd_name = c.get('name', '')
        for arg in c.get('arguments', []) or []:
            if arg and arg not in arg_names:
                missing_arg_instances.append(
                    f"Command \"{cmd_name}\" has argument \"{arg}\" but no Argument instance found")

    if missing_arg_instances:
        report.append("[Missing Argument Instances]")
        report.extend(f"- {msg}" for msg in missing_arg_instances)
        report.append("")

    # 3. Argument.command -> Command实例
    missing_cmd_refs = []
    for a in arguments:
        arg_name = a.get('name', '')
        cmd_ref = a.get('command')
        if cmd_ref and cmd_ref not in cmd_names:
            missing_cmd_refs.append(
                f"Argument \"{arg_name}\" references command \"{cmd_ref}\" but no Command instance found")

    if missing_cmd_refs:
        report.append("[Missing Command References]")
        report.extend(f"- {msg}" for msg in missing_cmd_refs)
        report.append("")

    # 4. Operation.related_commands -> Command/Parameter实例
    invalid_op_refs = []
    valid_refs = cmd_names | param_names
    for o in operations:
        op_name = o.get('name', '')
        for ref in o.get('related_commands', []) or []:
            if ref and ref not in valid_refs:
                invalid_op_refs.append(
                    f"Operation \"{op_name}\" references \"{ref}\" but not found in Command/Parameter")

    if invalid_op_refs:
        report.append("[Invalid Operation References]")
        report.extend(f"- {msg}" for msg in invalid_op_refs)
        report.append("")

    report.append("=== End Report ===")
    return '\n'.join(report)


def write_verification_report(filepath, commands, arguments, parameters, operations):
    """生成并写入验证报告文件"""
    report = verify_schemas(commands, arguments, parameters, operations)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    return report
