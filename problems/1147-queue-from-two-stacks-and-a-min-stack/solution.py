from typing import Any, List, Tuple


class QueueFromTwoStacks:
    """
    使用两个栈实现的 FIFO 队列。
    - in_stack: 仅负责接收新元素 (Push / Enqueue)
    - out_stack: 仅负责弹出旧元素 (Pop / Dequeue)
    """

    def __init__(self) -> None:
        self.in_stack: List[Any] = []
        self.out_stack: List[Any] = []

    def enqueue(self, val: Any) -> None:
        self.in_stack.append(val)

    def dequeue(self) -> Any:
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack.pop()

    def is_empty(self) -> bool:
        return not self.in_stack and not self.out_stack


class MinStack:
    """
    支持 O(1) 检索最小值的栈。
    维护 (val, min_so_far) 的元组栈，使得每个状态独立自洽，无需额外条件分支。
    """

    def __init__(self) -> None:
        self.stack: List[Tuple[Any, Any]] = []

    def push(self, val: Any) -> None:
        current_min = (
            min(val, self.stack[-1][1]) if self.stack else val
        )
        self.stack.append((val, current_min))

    def pop(self) -> Any:
        return self.stack.pop()[0]

    def top(self) -> Any:
        return self.stack[-1][0]

    def get_min(self) -> Any:
        return self.stack[-1][1]


def process_operations(operations: List[Tuple]) -> List[Any]:
    """处理混合操作的主入口函数。"""
    queue = QueueFromTwoStacks()
    min_stack = MinStack()
    outputs = []

    for op in operations:
        name = op[0]

        if name == "enqueue":
            queue.enqueue(op[1])
        elif name == "dequeue":
            outputs.append(queue.dequeue())
        elif name == "mpush":
            min_stack.push(op[1])
        elif name == "mpop":
            outputs.append(min_stack.pop())
        elif name == "mtop":
            outputs.append(min_stack.top())
        elif name == "mmin":
            outputs.append(min_stack.get_min())

    return outputs