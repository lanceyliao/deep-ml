import heapq

def compute_dilation_factor(tasks: list[dict], capacity: int) -> dict:
    """
    Computes the resource dilation factor for a constrained scheduling problem.

    Args:
        tasks: List of dicts with 'duration', 'resources', 'dependencies' keys.
        capacity: Total available resource units.

    Returns:
        Dict with 'dilation_factor', 'actual_makespan', 'lower_bound', 'start_times'.
    """
    n = len(tasks)
    if n == 0:
        return {
            "dilation_factor": 1.0,
            "actual_makespan": 0.0,
            "lower_bound": 0.0,
            "start_times": []
        }

    # ------------------------------------------------------------------
    # Step 1: Compute Early Finish (EF) & Critical Path Length
    # ------------------------------------------------------------------
    # Since dependencies form a DAG (indexed or topological order),
    # we compute Earliest Start (ES) and Earliest Finish (EF) for each task.
    es = [0.0] * n
    ef = [0.0] * n

    for i, task in enumerate(tasks):
        deps = task.get("dependencies", [])
        es[i] = max([ef[dep] for dep in deps], default=0.0)
        ef[i] = es[i] + task["duration"]

    critical_path = max(ef) if ef else 0.0

    # ------------------------------------------------------------------
    # Step 2: Compute Total Work & Theoretical Lower Bound
    # ------------------------------------------------------------------
    total_work = sum(t["duration"] * t["resources"] for t in tasks)
    work_lower_bound = total_work / capacity if capacity > 0 else float("inf")
    lower_bound = max(critical_path, work_lower_bound)

    # ------------------------------------------------------------------
    # Step 3: Discrete Event Simulation (Greedy List Scheduling)
    # ------------------------------------------------------------------
    # Track completion status and start times
    start_times = [None] * n
    completed = [False] * n
    
    # Pre-calculated in-degrees for dynamic updates
    in_degree = [len(t.get("dependencies", [])) for t in tasks]
    
    # Reverse dependency mapping: parent -> list of children
    dependents = [[] for _ in range(n)]
    for i, task in enumerate(tasks):
        for dep in task.get("dependencies", []):
            dependents[dep].append(i)

    # Priority queue storing ready task indices (prioritize by original index order)
    ready_queue = [i for i in range(n) if in_degree[i] == 0]
    heapq.heapify(ready_queue)

    # Discrete event heap: stores tuples of (completion_time, task_index)
    event_heap = []
    
    current_time = 0.0
    current_used_resources = 0

    while len(event_heap) > 0 or len(ready_queue) > 0:
        # Step A: Attempt to schedule ready tasks at current_time
        scheduled_any = True
        while scheduled_any and ready_queue:
            scheduled_any = False
            # Check candidate tasks in ready queue order
            temp_list = []
            while ready_queue:
                candidate = heapq.heappop(ready_queue)
                req_res = tasks[candidate]["resources"]
                
                if current_used_resources + req_res <= capacity:
                    # Allocate resources and schedule task
                    current_used_resources += req_res
                    start_times[candidate] = float(current_time)
                    finish_time = current_time + tasks[candidate]["duration"]
                    heapq.heappush(event_heap, (finish_time, candidate))
                    scheduled_any = True
                    break
                else:
                    # Resource constraint met; defer candidate for next iteration
                    temp_list.append(candidate)
            
            # Put unscheduled candidates back into the ready queue
            for item in temp_list:
                heapq.heappush(ready_queue, item)

        # Step B: Advance discrete clock to next completion event
        if event_heap:
            next_time, _ = event_heap[0]
            current_time = next_time
            
            # Process all tasks completing at current_time
            while event_heap and event_heap[0][0] == current_time:
                _, finished_task = heapq.heappop(event_heap)
                completed[finished_task] = True
                current_used_resources -= tasks[finished_task]["resources"]
                
                # Unlock dependent tasks
                for child in dependents[finished_task]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        heapq.heappush(ready_queue, child)

    # ------------------------------------------------------------------
    # Step 4: Calculate Actual Makespan & Resource Dilation Factor
    # ------------------------------------------------------------------
    actual_makespan = float(max((start_times[i] + tasks[i]["duration"] for i in range(n)), default=0.0))
    dilation_factor = actual_makespan / lower_bound if lower_bound > 0 else 1.0

    return {
        "dilation_factor": round(dilation_factor, 4),
        "actual_makespan": round(actual_makespan, 4),
        "lower_bound": round(lower_bound, 4),
        "start_times": [round(st, 4) for st in start_times]
    }