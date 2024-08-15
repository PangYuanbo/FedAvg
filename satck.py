class Stack:
    def __init__(self):
        self.stack = []

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        else:
            return None

    def top(self):
        if not self.is_empty():
            return self.stack[-1]
        else:
            return None

    def is_empty(self):
        return len(self.stack) == 0

    def __str__(self):
        return str(self.stack)

# 给定的数组 A
A = [1,22,13,84, 25, 6, 7, 58, 49, 1, 82, 43, 64, 5, 76, 57, 28, 69]

# 创建一个空栈 STK
STK = Stack()

# 按照给定的操作顺序进行操作
STK.push(A[0])
STK.push(A[1])
STK.push(A[2])
STK.pop()
STK.push(A[3])
STK.top()
STK.top()
STK.push(A[4])
STK.push(A[5])
STK.pop()
STK.push(A[6])
STK.top()
STK.top()
STK.push(A[7])
STK.push(A[8])
STK.pop()
STK.push(A[9])

# 打印栈中元素从顶到底的顺序
print("Stack from top to tail:", STK)

# 打印栈中元素的值从顶到底的顺序
print("Values in the stack from top to tail:", STK.stack)
