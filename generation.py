import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

from openai import OpenAI


SCENES = [
    "office work",
    "video editing",
    "report writing",
    "online courses",
    "gaming",
    "socializing",
    "creative writing",
    "entertainment",
    "travel planning",
    "programming and development",
    "event planning",
    "data analysis",
    "education and research",
    "career development",
    "health and fitness",
    "language learning",
    "decision making",
    "content creation",
    "project management",
    "personal organization",
    "time management",
    "personal finance",
    "shopping",
]


def prepare_prompt(
    tasks: List[Dict[str, Any]],
    seed_num: int,
    rng_seed: Optional[int] = 42,
    min_key_num: int = 10,
    max_key_num: int = 12,
) -> str:
    if seed_num > len(tasks):
        raise ValueError(f"seed_num ({seed_num}) cannot exceed len(tasks) ({len(tasks)})")
    
    if max_key_num < min_key_num:
        raise ValueError("Invalid key number range")
    
    rng = random.Random(rng_seed)
    seed_tasks = rng.sample(tasks, k=seed_num)
    
    scene = rng.choice(SCENES) if SCENES else "office work"
    min_items = rng.randint(min_key_num, max_key_num)

    prompt_parts: List[str] = []
    prompt_parts.append(
        "You are a data generator for personalized user queries.\n"
        "You will be given SEED examples. Each example contains:\n"
        "- Query\n"
        "- Memories (JSON)\n"
        "- User Portrait\n\n"
        "Goal: generate 1 NEW sample that follow the SAME format as the seeds.\n"
        "The new samples must be diverse and realistic.\n"
        f"SCENE (must follow): {scene}\n\n"
        "CRITICAL RULE:\n"
        "- The 'User Portrait' MUST be inferred from the generated 'Query' and the included 'Memories (JSON)', but it MUST NOT explicitly mention, quote, or cite the query or the memories\n"
        "- Do NOT invent portrait traits that are not supported by the query/memories.\n"
        "- The string value of 'Query' and 'User Portrait' MUST be plain natural language only. They MUST NOT contain any keys/labels/field headers or JSON snippets.\n\n"
        "Diversity requirements:\n"
        "- Vary intent types (planning, troubleshooting, summarization, drafting, rewriting, comparison, automation).\n"
        "- Vary constraints (deadline, device/OS, tools, file formats, audience, length).\n"
        "- Vary skill level and preferences (beginner vs advanced, concise vs detailed).\n"
        "- Avoid copying seed wording; do not reuse the same named entities unless justified by memories.\n\n"
        "Output requirements (STRICT):\n"
        "- Output MUST be a SINGLE JSON object (and nothing else).\n"
        "- The JSON object MUST have EXACT keys:\n"
        '   ["query", "user_portrait", "memories"]\n'
        f"  'memories' MUST be a JSON array of objects (not strings). Include at least {min_items} items.\n"
        "  'user_portrait' MUST be a natural-language paragraph, and it should reflect/cover as many of the generated memories as possible. It should be slightly longer than the seed portraits, but not too long: aim for about 8-12 sentences."
        "- The memory objects should follow the same structure as the seed memory JSON you see below.\n\n"
        "SEED EXAMPLES (each seed is separated by a delimiter for readability):\n"
    )

    for i, task in enumerate(seed_tasks, start=1):
        prompt_parts.append(f"------------- Example {i} ------------------\n")
        prompt_parts.append(f"Query: {task.get('query')}\n")
        prompt_parts.append("Memories:\n")
        for mem in task.get("matched_memories", []):
            if mem.get("found") and mem.get("memory_json_str"):
                prompt_parts.append(f"{mem['memory_json_str']}\n")
        prompt_parts.append(f"User Portrait: {task.get('user_portrait')}\n")
        prompt_parts.append("\n")

    prompt_parts.append(
        "NOW GENERATE 1 NEW SAMPLE.\n"
        "Remember: output ONLY one JSON object that satisfies the requirements.\n"
        "Reminder: the text inside `query` and `user_portrait` must contain NO key names, NO labels, NO JSON fragments.\n"
    )

    return "".join(prompt_parts)


def extract_ltm_keys_from_memories(memories: Any) -> List[str]:
    keys: List[str] = []
    if not isinstance(memories, list):
        return keys

    for memory in memories:
        if not isinstance(memory, dict):
            continue
        j_data = memory.get("j_data")
        if isinstance(j_data, dict):
            key = j_data.get("key")
            if isinstance(key, str) and key:
                keys.append(key)
                continue
    
    seen = set()
    out = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            out.append(key)

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Personalized Query Generation")

    # Model configs
    parser.add_argument("--model", type=str, default="qwen3-max", help="Model name")
    parser.add_argument("--api_key", type=str, default="sk-e8401b009e4b4d3d97005fb731979db9", help="API key")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="Base URL for the API")

    # Data configs
    parser.add_argument("--data_path", type=str, default="./data/memories_and_queries.json", help="Path to load data")
    parser.add_argument("--output", type=str, default="./outputs/", help="Path to save the results")

    # Sampling configs
    parser.add_argument("--max_tokens", type=int, default=32768, help="Maximum number of new tokens the model is allowed to generate per response")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Controls the cumulative probability of the top tokens to consider")
    parser.add_argument("--top_k", type=int, default=0, help="Controls the number of top tokens to consider")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"], help="Precision dtype")

    # Generation configs
    parser.add_argument("--seed_num", type=int, default=3, help="The number of seed data")
    parser.add_argument("--user_num", type=int, default=400, help="The number of users")
    parser.add_argument("--sample_num", type=int, default=6, help="The number of new data to generate")
    parser.add_argument("--min_key_num", type=int, default=10, help="The minimum number of ltm_keys required in the generated memories")
    parser.add_argument("--max_key_num", type=int, default=12, help="The maximum number of ltm_keys allowed in the generated memories")
    parser.add_argument("--base_id", type=int, default=40, help="The base id to start")
    parser.add_argument("--max_retries", type=int, default=15, help="The max retries per task")
    
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at: {args.data_path}")
    
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    # Load data
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("Data file is empty or invalid")

    # Initialize OpenAI client
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )
    if not client:
        raise ValueError("Failed to create OpenAI client")

    taskandltmkey = data.get("taskandltmkey", [])
    memories = data.get("memories", [])
    if not taskandltmkey or not memories:
        raise ValueError("Data file must contain 'taskandltmkey' and 'memories' fields with non-empty lists")
    
    # Collect seed data
    all_tasks_out: List[Dict[str, Any]] = []
    for user_item in taskandltmkey:
        user_id = user_item.get("user_id")
        tasks = user_item.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("The format of 'tasks' is invalid")
    
        for task in tasks:
            task_id = task.get("task_id")
            query = task.get("query")
            user_portrait = task.get("user_portrait")
            ltm_keys = task.get("ltm_keys", [])
            if not isinstance(ltm_keys, list) or not ltm_keys:
                raise ValueError("The format of 'ltm_keys' is invalid")
            
            matched_memories = []
            for key in ltm_keys:
                if not isinstance(key, str):
                    continue
                
                mem_obj = next(
                    (
                        m for m in memories
                        if isinstance(m, dict)
                        and isinstance(m.get("j_data"), dict)
                        and m["j_data"].get("key") == key
                    ),
                    None
                )

                if mem_obj is None:
                    matched_memories.append({
                        "ltm_key": key,
                        "found": False,
                        "memory_json_str": None,
                    })
                else:
                    try:
                        mem_json_str = json.dumps(mem_obj, ensure_ascii=False, indent=2)
                        matched_memories.append({
                            "ltm_key": key,
                            "found": True,
                            "memory_json_str": mem_json_str,
                        })
                    except Exception as e:
                        matched_memories.append({
                            "ltm_key": key,
                            "found": False,
                            "memory_json_str": None,
                        })
            
            all_tasks_out.append({
                "user_id": user_id,
                "task_id": task_id,
                "query": query,
                "user_portrait": user_portrait,
                "ltm_keys": ltm_keys,
                "matched_memories": matched_memories,
            })

    # Build a set of existing ltm_keys to avoid duplicates
    existing_ltm_keys = set()
    for t in all_tasks_out:
        for k in t.get("ltm_keys", []):
            if isinstance(k, str):
                existing_ltm_keys.add(k)

    # Start generation loop
    for user_id in range(args.user_num):
        new_user_id = args.base_id + user_id

        current_user_tasks = []

        print(f"Generating for user: {new_user_id}...")

        for sample_id in range(args.sample_num):
            try_count = 0
            while True:
                if try_count >= args.max_retries:
                    print(f"Task {sample_id}: Failed after {args.max_retries} retries, skipping")
                    break
                
                # Prepare prompt
                input = prepare_prompt(
                    all_tasks_out,
                    args.seed_num,
                    rng_seed=(new_user_id * 1000 + sample_id * 100 + try_count),
                    min_key_num=args.min_key_num,
                    max_key_num=args.max_key_num,
                )

                # Generate response
                completion = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "user", "content": input},
                    ],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    response_format={"type": "json_object"}
                )

                output_str = completion.choices[0].message.content

                # Parse and validate output
                try:
                    output_json = json.loads(output_str)
                except json.JSONDecodeError:
                    try_count += 1
                    print(f"Task {sample_id}: JSON decode error, retrying... (attempt {try_count}/{args.max_retries})")
                    continue
                    
                if not (
                    isinstance(output_json, dict)
                    and "query" in output_json
                    and "user_portrait" in output_json
                    and "memories" in output_json
                ):
                    try_count += 1
                    print(f"Task {sample_id}: Output JSON missing required keys, retrying... (attempt {try_count}/{args.max_retries})")
                    continue
                
                query = output_json["query"]
                user_portrait = output_json["user_portrait"]
                generated_memories = output_json["memories"]
                raw_ltm_keys = extract_ltm_keys_from_memories(generated_memories)
                if len(raw_ltm_keys) < args.min_key_num:
                    try_count += 1
                    print(f"Task {sample_id}: Not enough ltm_keys ({len(raw_ltm_keys)} found, \
                          minimum is {args.min_key_num}), retrying... (attempt {try_count}/{args.max_retries})")
                    continue

                final_ltm_keys: List[str] = []
                rename_map: Dict[str, str] = {}
                for key in raw_ltm_keys:
                    original_key = key
                    new_key = key
                    if new_key in existing_ltm_keys:
                        new_key = f"{original_key}_{new_user_id}_{sample_id}"
                        rename_map[original_key] = new_key
                    
                    existing_ltm_keys.add(new_key)
                    final_ltm_keys.append(new_key)

                if isinstance(generated_memories, list) and rename_map:
                    for memory in generated_memories:
                        if not isinstance(memory, dict):
                            continue
                        j_data = memory.get("j_data")
                        if isinstance(j_data, dict):
                            key = j_data.get("key")
                            if isinstance(key, str) and key in rename_map:
                                j_data["key"] = rename_map[key]

                memories.extend(generated_memories)
                
                new_task_obj = {
                    "task_id": sample_id,
                    "query": query,
                    "ltm_keys": final_ltm_keys,
                    "user_portrait": user_portrait
                }
                current_user_tasks.append(new_task_obj)

                new_matched_memories = []
                for key in final_ltm_keys:
                    found_memory = next((m for m in generated_memories 
                                      if (m.get("j_data", {}).get("key") == key or m.get("key") == key)), None)

                    memory_str = None
                    if found_memory:
                        memory_str = json.dumps(found_memory, ensure_ascii=False, indent=2)
                    
                    new_matched_memories.append({
                        "ltm_key": key,
                        "found": True if found_memory else False,
                        "memory_json_str": memory_str
                    })
                
                all_tasks_out.append({
                    "user_id": new_user_id,
                    "task_id": sample_id,
                    "query": query,
                    "user_portrait": user_portrait,
                    "ltm_keys": final_ltm_keys,
                    "matched_memories": new_matched_memories
                })

                print(f"Task {sample_id}: Generated successfully")
                break

        if current_user_tasks:
            taskandltmkey.append({
                "user_id": new_user_id,
                "tasks": current_user_tasks
            })
    
    # Save results
    output_file = os.path.join(args.output, "memories_and_queries.json")
    print(f"Saving results to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)