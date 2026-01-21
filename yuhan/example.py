#!/usr/bin/env python3
"""Example: 3-stage schema extraction from EDA document"""

import json
from openai import OpenAI
from prompt import (
    STAGE1_IDENTIFY_CLASSES, STAGE1_GLEAN, STAGE2_PROMPTS,
    verify_schemas, write_verification_report
)

# client = OpenAI()
# model = "gpt-4"
host = "http://120.25.194.125:2105/v1"
apikey = "xxx"
model = "qwen_docqa"

client = OpenAI(
    base_url=host,
    api_key=apikey,
    timeout=300.0
)
MAX_GLEAN_ROUNDS = 3


def call_llm(prompt):
    """Call LLM and return JSON response"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content)


def stage1_identify(document, notes=""):
    """Stage 1: Identify instances with glean"""
    # Initial extraction
    prompt = STAGE1_IDENTIFY_CLASSES.format(document=document, document_notes=notes)
    result = call_llm(prompt)
    all_instances = result.get("identified_classes", {})

    # Glean rounds
    for _ in range(MAX_GLEAN_ROUNDS):
        prompt = STAGE1_GLEAN.format(
            previous_results=json.dumps(all_instances, ensure_ascii=False),
            document=document
        )
        glean = call_llm(prompt).get("additional_instances", {})

        # Merge new instances
        added = False
        for cls, instances in glean.items():
            if instances:
                all_instances.setdefault(cls, []).extend(instances)
                added = True
        if not added:
            break

    return all_instances


def stage2_extract(document, identified):
    """Stage 2: Extract details for each instance"""
    extracted = {cls: [] for cls in identified}

    for cls, instances in identified.items():
        if cls not in STAGE2_PROMPTS:
            continue
        for instance in instances:
            prompt = STAGE2_PROMPTS[cls].format(instance=instance, document=document)
            data = call_llm(prompt)
            extracted[cls].append(data)

    return extracted


def stage3_verify(extracted, output_file="verification_report.txt"):
    """Stage 3: Verify and generate report"""
    return write_verification_report(
        output_file,
        commands=extracted.get("Command", []),
        arguments=extracted.get("Argument", []),
        parameters=extracted.get("Parameter", []),
        operations=extracted.get("Operation", [])
    )


def main():
    # Sample document
    doc = """
    fix_setup_path_violations命令用于修复setup timing违规。
    语法: fix_setup_path_violations [-size_down_only] [-force]
    -size_down_only: 只对cell进行size down操作
    -force: 强制执行修复
    """

    # Stage 1
    print("=== Stage 1: Identify ===")
    identified = stage1_identify(doc)
    print(json.dumps(identified, indent=2, ensure_ascii=False))

    # Stage 2
    print("\n=== Stage 2: Extract ===")
    extracted = stage2_extract(doc, identified)
    print(json.dumps(extracted, indent=2, ensure_ascii=False))

    # Stage 3
    print("\n=== Stage 3: Verify ===")
    report = stage3_verify(extracted)
    print(report)


if __name__ == "__main__":
    main()
